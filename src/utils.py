import os
import sys
import json
import copy
import math
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import logging
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset, load_from_disk, DatasetDict
from torch.utils.data import DataLoader, random_split
from IPython import get_ipython
from IPython.display import display, HTML

logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

_TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2")
_TOKENIZER.pad_token = _TOKENIZER.eos_token
_LAST_PROGRESS_MESSAGE_LEN = 0
_ON_PROGRESS_LINE = False
_LAST_PPO_PROGRESS_LINES = 0
_PPO_PROGRESS_HANDLE = None

def ensure_newline():
    global _ON_PROGRESS_LINE, _LAST_PPO_PROGRESS_LINES, _LAST_PROGRESS_MESSAGE_LEN, _PPO_PROGRESS_HANDLE
    _PPO_PROGRESS_HANDLE = None
    if _ON_PROGRESS_LINE:
        sys.stdout.write("\n")
        sys.stdout.flush()
        _ON_PROGRESS_LINE = False
        _LAST_PPO_PROGRESS_LINES = 0
        _LAST_PROGRESS_MESSAGE_LEN = 0

def load_data(max_length = 512, overwrite = False):
    save_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "../data"))
    exists = os.path.exists(os.path.join(save_path, "dataset_dict.json"))
    if not overwrite and exists:
        return load_from_disk(save_path)
    dataset = load_dataset("Anthropic/hh-rlhf")

    def process(example):
        prompt = extract_prompt(example["chosen"])
        prompt_len = len(_TOKENIZER(prompt, truncation = False)["input_ids"])
        chosen_len = len(_TOKENIZER(example["chosen"], truncation = False)["input_ids"])
        rejected_len = len(_TOKENIZER(example["rejected"], truncation = False)["input_ids"])
        return {
            "prompt_len": prompt_len,
            "chosen_len": chosen_len,
            "rejected_len": rejected_len,
            "keep": chosen_len <= max_length and rejected_len <= max_length,
        }
    
    dataset = dataset.map(process)
    dataset = dataset.filter(lambda x: x["keep"])
    dataset = dataset.remove_columns(["keep"])
    print(f"median prompt length:   {int(np.median(dataset['train']['prompt_len']))} tokens")
    print(f"median chosen length:   {int(np.median(dataset['train']['chosen_len']))} tokens")
    print(f"median rejected length: {int(np.median(dataset['train']['rejected_len']))} tokens")
    dataset = dataset.remove_columns(["prompt_len", "chosen_len", "rejected_len"])
    os.makedirs(save_path, exist_ok = True)
    dataset.save_to_disk(save_path)
    return dataset

def extract_prompt(text):
    idx = text.rfind("\n\nAssistant:")
    if idx == -1:
        return text
    return text[:idx + len("\n\nAssistant:")]

def show_progress(bi, total_batches, epoch = None, grad_norm = None):
    global _LAST_PROGRESS_MESSAGE_LEN, _ON_PROGRESS_LINE
    pct = 100.0 * bi / max(1, total_batches)
    parts = [f"epoch {epoch}: {pct:6.2f}%" if epoch is not None else f"progress: {pct:6.2f}%"]
    if grad_norm is not None:
        parts.append(f"grad: {grad_norm:6.4g}")
    msg = "   ".join(parts)
    padding = " " * max(0, _LAST_PROGRESS_MESSAGE_LEN - len(msg))
    sys.stdout.write("\r" + msg + padding)
    sys.stdout.flush()
    _LAST_PROGRESS_MESSAGE_LEN = len(msg)
    _ON_PROGRESS_LINE = True

def show_progress_ppo(bi, total_batches, epoch, ppo_step, ppo_epochs, reward, kl, beta, drift, entropy, grad_norm_policy, grad_norm_value):
    global _LAST_PPO_PROGRESS_LINES, _ON_PROGRESS_LINE, _PPO_PROGRESS_HANDLE
    pct = 100.0 * bi / max(1, total_batches)
    lines = [
        f"epoch {epoch}   batch {bi}/{total_batches}   step {ppo_step + 1}/{ppo_epochs}   {pct:.1f}%",
        f"  reward: {reward:.3f}   drift: {drift:+.4f}   entropy: {entropy:.4f}",
        f"  policy grad: {grad_norm_policy:.4g}   value grad: {grad_norm_value:.4g}",
        f"  KL: {kl:.4f}   beta: {beta:.4f}",
    ]
    if get_ipython() is not None:
        html = HTML("<pre style='margin:0;line-height:1.4'>" + "\n".join(lines) + "</pre>")
        if _PPO_PROGRESS_HANDLE is None:
            _PPO_PROGRESS_HANDLE = display(html, display_id=True)
        else:
            _PPO_PROGRESS_HANDLE.update(html)
        _LAST_PPO_PROGRESS_LINES = len(lines)
        return
    # terminal fallback
    if _LAST_PPO_PROGRESS_LINES > 0:
        sys.stdout.write("\r" + "\x1b[A\r" * (_LAST_PPO_PROGRESS_LINES - 1))
    for i, line in enumerate(lines):
        sys.stdout.write(line + "\x1b[K")
        if i < len(lines) - 1:
            sys.stdout.write("\n")
    sys.stdout.flush()
    _LAST_PPO_PROGRESS_LINES = len(lines)
    _ON_PROGRESS_LINE = True

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model file not found: {model_path}")
    model = torch.load(model_path, map_location = "cpu", weights_only = False)
    return model

def load_history(*history_paths):
    history = {}
    for path in history_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"history file not found: {path}")
        h = torch.load(path, map_location = "cpu")
        for key, values in h.items():
            if key not in history:
                history[key] = []
            history[key].extend(values)
    return history

def plot_history(history, log_x = False, log_y = False, batches_per_epoch = None, title = "Training History", keys = None):
    if keys is not None:
        for k in keys:
            if k not in history:
                print(f"warning: key '{k}' not found in history")
        plot_keys = [k for k in keys if k in history]
    else:
        plot_keys = list(history.keys())
    total_batches = max((len(history[k]) for k in plot_keys), default = 0)
    plt.figure(figsize = (10, 5))
    for key in plot_keys:
        series = history[key]
        xs = [i for i, v in enumerate(series) if np.isfinite(v)]
        ys = [v for v in series if np.isfinite(v)]
        if xs:
            plt.plot(xs, ys, label = key)
    if batches_per_epoch is not None and batches_per_epoch > 0:
        k = 1
        while True:
            x = k * batches_per_epoch
            if x > total_batches:
                break
            if not (log_x and x == 0):
                plt.axvline(x = x, linestyle = "--", linewidth = 1, alpha = 0.5)
            k += 1
    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")
    plt.xlabel("Batch")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ==============================================================================
# 1 - Reward Model
# ==============================================================================

class RewardDataset(Dataset):
    def __init__(self, ds, tokenizer = _TOKENIZER, max_length = 512):
        self.ds = ds
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        chosen = self.ds[idx]["chosen"]
        rejected = self.ds[idx]["rejected"]
        chosen_enc = self.tokenizer(
            chosen, 
            truncation = True,
            max_length = self.max_length,
            padding = "max_length",
            return_tensors = "pt"
        )
        rejected_enc = self.tokenizer(
            rejected,
            truncation = True,
            max_length = self.max_length,
            padding = "max_length",
            return_tensors = "pt"
        )
        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
        }
    
class RewardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = GPT2Model.from_pretrained("gpt2")
        self.backbone.resize_token_embeddings(len(_TOKENIZER))
        self.score = torch.nn.Linear(self.backbone.config.hidden_size, 1, bias = False)

    def forward(self, input_ids, attention_mask):
        B = input_ids.shape[0]
        embeddings = self.backbone(input_ids = input_ids, attention_mask = attention_mask).last_hidden_state
        positions = torch.arange(input_ids.shape[1], device = input_ids.device).unsqueeze(0)
        sequence_lengths = (attention_mask * positions).argmax(dim = 1)
        last_token_embeddings = embeddings[torch.arange(B, device = input_ids.device), sequence_lengths]
        return self.score(last_token_embeddings).squeeze(-1)
    
def prepare_reward_dataloaders(dataset, batch_size = 8, subset_fraction = 0.1, val_fraction = 0.1, max_length = 512, seed = 42):
    generator = torch.Generator().manual_seed(seed)
    train_split = dataset["train"]
    subset_size = int(len(train_split) * subset_fraction)
    indices = torch.randperm(len(train_split), generator = generator)[:subset_size].tolist()
    ds = RewardDataset(train_split.select(indices), _TOKENIZER, max_length = max_length)
    val_size = int(len(ds) * val_fraction)
    train_ds, valid_ds = random_split(ds, [len(ds) - val_size, val_size], generator = generator)
    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
    valid_loader = DataLoader(valid_ds, batch_size = batch_size, shuffle = False)
    return train_loader, valid_loader

def train_reward_model(
    model,
    train_loader,
    valid_loader,
    epochs,
    model_dir = "../model/reward",
    dropout = 0.1,
    lr = 2e-5,
    weight_decay = 0.01,
    warmup_steps = 200,
    max_grad_norm = 25
):
    global _LAST_PROGRESS_MESSAGE_LEN
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    os.makedirs(model_dir, exist_ok = True)
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model = model.to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "weight_decay": 0.0},
        {"params": model.score.parameters(), "weight_decay": weight_decay},
    ], lr = lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: min(1.0, step / max(1, warmup_steps)))
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = dropout
    total_train_batches = len(train_loader)
    total_valid_batches = len(valid_loader)
    if total_valid_batches > 0:
        valid_every = max(1, int(round(total_train_batches / total_valid_batches)))
    else:
        valid_every = None
    if epochs == 0:
        return model
    for epoch in range(1, epochs + 1):
        _LAST_PROGRESS_MESSAGE_LEN = 0
        model.train()
        train_batch_losses = []
        valid_batch_losses = []
        clipped_batches = []
        valid_iter = iter(valid_loader) if valid_every is not None else None
        show_progress(0, total_train_batches, epoch = epoch)
        for bi, batch in enumerate(train_loader, start = 1):
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)
            chosen_scores = model(chosen_ids, chosen_mask)
            rejected_scores = model(rejected_ids, rejected_mask)
            loss = -F.logsigmoid(chosen_scores - rejected_scores).mean()
            loss_value = float(loss.item())
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm if bi > warmup_steps else float("inf"))
            grad_norm_value = float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm)
            if bi > warmup_steps and grad_norm_value > max_grad_norm:
                sys.stdout.write("\n")
                sys.stdout.flush()
                clipped_batches.append({
                    "epoch": epoch,
                    "batch_index": bi,
                    "loss": loss_value,
                    "grad_norm": grad_norm_value,
                })
                print(f"gradient clipped: epoch {epoch}, batch {bi}: norm = {grad_norm_value:.4g}")
            optimizer.step()
            scheduler.step()
            train_batch_losses.append(loss_value)
            valid_batch_losses.append(float("nan"))
            if valid_iter is not None and (bi % valid_every == 0):
                model.eval()
                with torch.no_grad():
                    try:
                        vbatch = next(valid_iter)
                    except StopIteration:
                        valid_iter = iter(valid_loader)
                        vbatch = next(valid_iter)
                    vc_ids = vbatch["chosen_input_ids"].to(device)
                    vc_mask = vbatch["chosen_attention_mask"].to(device)
                    vr_ids = vbatch["rejected_input_ids"].to(device)
                    vr_mask = vbatch["rejected_attention_mask"].to(device)
                    vc_scores = model(vc_ids, vc_mask)
                    vr_scores = model(vr_ids, vr_mask)
                    vloss = -F.logsigmoid(vc_scores - vr_scores).mean()
                valid_batch_losses[-1] = float(vloss.item())
                model.train()
            display_grad_norm = min(grad_norm_value, max_grad_norm) if bi > warmup_steps else grad_norm_value
            show_progress(bi, total_train_batches, epoch = epoch, grad_norm = display_grad_norm)
        sys.stdout.write("\n")
        sys.stdout.flush()
        history = {"train": train_batch_losses, "valid": valid_batch_losses}
        tag = f"e{epoch}"
        epoch_dir = os.path.join(model_dir, f"{run_id}-{tag}")
        os.makedirs(epoch_dir, exist_ok = True)
        torch.save(model, os.path.join(epoch_dir, f"model-{run_id}-{tag}.pt"))
        torch.save(history, os.path.join(epoch_dir, f"history-{run_id}-{tag}.pt"))
        bad_path = os.path.join(epoch_dir, f"bad_batches-{run_id}-{tag}.json")
        with open(bad_path, "w", encoding = "utf-8") as f:
            json.dump(clipped_batches, f, ensure_ascii = False, indent = 2)
    return model
    
# ==============================================================================
# 2 — Supervised Fine-Tuning (SFT)
# ==============================================================================

class SFTDataset(Dataset):
    def __init__(self, ds, tokenizer = _TOKENIZER, max_length = 512):
        self.ds = ds
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        chosen = self.ds[idx]["chosen"]
        enc = self.tokenizer(
            chosen,
            truncation = True,
            max_length = self.max_length,
            padding = "max_length",
            return_tensors = "pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        prompt_enc = self.tokenizer(
            extract_prompt(chosen),
            truncation = True,
            max_length = self.max_length,
            return_tensors = "pt"
        )
        labels[:prompt_enc["input_ids"].shape[1]] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
def prepare_sft_dataloaders(dataset, batch_size = 8, subset_fraction = 0.1, val_fraction = 0.1, max_length = 512, seed = 42):
    generator = torch.Generator().manual_seed(seed)
    train_split = dataset["train"]
    subset_size = int(len(train_split) * subset_fraction)
    indices = torch.randperm(len(train_split), generator = generator)[:subset_size].tolist()
    ds = SFTDataset(train_split.select(indices), _TOKENIZER, max_length = max_length)
    val_size = int(len(ds) * val_fraction)
    train_ds, valid_ds = random_split(ds, [len(ds) - val_size, val_size], generator = generator)
    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
    valid_loader = DataLoader(valid_ds, batch_size = batch_size, shuffle = False)
    return train_loader, valid_loader

def train_sft_model(
    model,
    train_loader,
    valid_loader,
    epochs,
    model_dir = "../model/sft",
    lr = 2e-5,
    weight_decay = 0.01,
    warmup_steps = 200,
    max_grad_norm = 15
):
    global _LAST_PROGRESS_MESSAGE_LEN
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    os.makedirs(model_dir, exist_ok = True)
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model = model.to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.transformer.parameters(), "weight_decay": 0.0},
        {"params": [p for p in model.lm_head.parameters() if not any(p is q for q in model.transformer.parameters())], "weight_decay": weight_decay},
    ], lr = lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: min(1.0, step / max(1, warmup_steps)))
    total_train_batches = len(train_loader)
    total_valid_batches = len(valid_loader)
    valid_every = max(1, int(round(total_train_batches / total_valid_batches))) if total_valid_batches > 0 else None
    if epochs == 0:
        return model
    for epoch in range(1, epochs + 1):
        _LAST_PROGRESS_MESSAGE_LEN = 0
        model.train()
        train_batch_losses = []
        valid_batch_losses = []
        clipped_batches = []
        valid_iter = iter(valid_loader) if valid_every is not None else None
        show_progress(0, total_train_batches, epoch = epoch)
        for bi, batch in enumerate(train_loader, start = 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
            loss = outputs.loss
            loss_value = float(loss.item())
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm if bi > warmup_steps else float("inf")
            )
            grad_norm_value = float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm)
            if bi > warmup_steps and grad_norm_value > max_grad_norm:
                sys.stdout.write("\n")
                sys.stdout.flush()
                clipped_batches.append({
                    "epoch": epoch,
                    "batch_index": bi,
                    "loss": loss_value,
                    "grad_norm": grad_norm_value,
                })
                print(f"gradient clipped: epoch {epoch}, batch {bi}: norm = {grad_norm_value:.4g}")
            optimizer.step()
            scheduler.step()
            train_batch_losses.append(loss_value)
            valid_batch_losses.append(float("nan"))
            if valid_iter is not None and (bi % valid_every == 0):
                model.eval()
                with torch.no_grad():
                    try:
                        vbatch = next(valid_iter)
                    except StopIteration:
                        valid_iter = iter(valid_loader)
                        vbatch = next(valid_iter)
                    v_ids = vbatch["input_ids"].to(device)
                    v_mask = vbatch["attention_mask"].to(device)
                    v_labels = vbatch["labels"].to(device)
                    voutputs = model(input_ids = v_ids, attention_mask = v_mask, labels = v_labels)
                    valid_batch_losses[-1] = float(voutputs.loss.item())
                model.train()
            display_grad_norm = min(grad_norm_value, max_grad_norm) if bi > warmup_steps else grad_norm_value
            show_progress(bi, total_train_batches, epoch = epoch, grad_norm = display_grad_norm)
        sys.stdout.write("\n")
        sys.stdout.flush()
        history = {"train": train_batch_losses, "valid": valid_batch_losses}
        tag = f"e{epoch}"
        epoch_dir = os.path.join(model_dir, f"{run_id}-{tag}")
        os.makedirs(epoch_dir, exist_ok = True)
        torch.save(model, os.path.join(epoch_dir, f"model-{run_id}-{tag}.pt"))
        torch.save(history, os.path.join(epoch_dir, f"history-{run_id}-{tag}.pt"))
        with open(os.path.join(epoch_dir, f"bad_batches-{run_id}-{tag}.json"), "w", encoding = "utf-8") as f:
            json.dump(clipped_batches, f, ensure_ascii = False, indent = 2)
    return model
    
# ==============================================================================
# 3 — Reinforcement Learning from Human Feedback (PPO)
# ==============================================================================

class PPODataset(Dataset):
    def __init__(self, ds, tokenizer = _TOKENIZER, max_length = 512):
        self.ds = ds
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        prompt = extract_prompt(self.ds[idx]["chosen"])
        enc = self.tokenizer(
            prompt,
            truncation = True,
            max_length = self.max_length,
            padding = "max_length",
            return_tensors = "pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0)
        }
    
class ValueModel(torch.nn.Module):
    def __init__(self, sft_model):
        super().__init__()
        self.backbone = copy.deepcopy(sft_model.transformer)
        self.value_head = torch.nn.Linear(self.backbone.config.hidden_size, 1, bias = False)
        torch.nn.init.zeros_(self.value_head.weight)

    def forward(self, input_ids, attention_mask):
        embeddings = self.backbone(input_ids = input_ids, attention_mask = attention_mask).last_hidden_state
        return self.value_head(embeddings).squeeze(-1)  # (B, T) per-token values

def prepare_ppo_dataloader(dataset, batch_size = 8, subset_fraction = 0.1, seed = 42):
    generator = torch.Generator().manual_seed(seed)
    train_split = dataset["train"]
    subset_size = int(len(train_split) * subset_fraction)
    indices = torch.randperm(len(train_split), generator = generator)[:subset_size].tolist()
    ds = PPODataset(train_split.select(indices), _TOKENIZER)
    return DataLoader(ds, batch_size = batch_size, shuffle = True, generator = generator)

def compare_models(dataset, reward_model, reference_model, policy_model, max_samples = 800, batch_size = 8, seed = 42):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    generator = torch.Generator().manual_seed(seed)
    reward_model = reward_model.to(device).eval()
    reference_model = reference_model.to(device).eval()
    policy_model = policy_model.to(device).eval()
    test_split = dataset["test"]
    n = min(max_samples, len(test_split))
    indices = torch.randperm(len(test_split), generator = generator)[:n].tolist()
    ds = PPODataset(test_split.select(indices), _TOKENIZER)
    loader = DataLoader(ds, batch_size = batch_size, shuffle = False)
    reference_rewards, policy_rewards = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ref_ids = reference_model.generate(
                input_ids = input_ids, attention_mask = attention_mask,
                max_new_tokens = 160, do_sample = True, temperature = 0.9,
                pad_token_id = _TOKENIZER.eos_token_id
            )
            pol_ids = policy_model.generate(
                input_ids = input_ids, attention_mask = attention_mask,
                max_new_tokens = 160, do_sample=True, temperature = 0.9,
                pad_token_id = _TOKENIZER.eos_token_id
            )
            ref_mask = (ref_ids != _TOKENIZER.pad_token_id).long()
            pol_mask = (pol_ids != _TOKENIZER.pad_token_id).long()
            reference_rewards.extend(reward_model(ref_ids, ref_mask).tolist())
            policy_rewards.extend(reward_model(pol_ids, pol_mask).tolist())

    ref_arr = np.array(reference_rewards)
    pol_arr = np.array(policy_rewards)
    t_stat, _ = stats.ttest_rel(pol_arr, ref_arr)

    print(f"Reference: {ref_arr.mean():.4f} ± {ref_arr.std():.4f}")
    print(f"Policy:    {pol_arr.mean():.4f} ± {pol_arr.std():.4f}")
    print(f"paired t-test: t = {t_stat:.3f}")

def log_probs(model, ids, mask, prompt_lengths = None):
    logits = model(input_ids = ids, attention_mask = mask).logits
    lp = F.log_softmax(logits[:, :-1, :], dim = -1)
    token_log_prob = lp.gather(2, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    token_mask = mask[:, 1:].float()
    if prompt_lengths is not None:
        positions = torch.arange(1, ids.shape[1], device = ids.device).unsqueeze(0)
        token_mask = token_mask * (positions >= prompt_lengths.unsqueeze(1)).float()
    return token_log_prob * token_mask

def compute_gae(values, rewards, response_mask, lam = 0.95):
    device = values.device
    B, T = values.shape[0], values.shape[1] - 1
    v_curr = values[:, :-1]
    v_next = values[:, 1:]

    # Last response position per sequence (index into T-1 space)
    count = response_mask.cumsum(dim = 1) * response_mask
    has_response = (response_mask.sum(dim = 1) > 0)
    last_k = count.argmax(dim = 1).clamp(max = T - 1)  # (B,)
    is_terminal = (torch.arange(T, device = device).unsqueeze(0) == last_k.unsqueeze(1)).float()
    is_terminal = is_terminal * has_response.float().unsqueeze(1)

    # δ_t = r_t + V_{t+1} - V_t
    delta = (rewards.unsqueeze(1) * is_terminal + (1 - is_terminal) * v_next - v_curr) * response_mask
    gae = torch.zeros(B, device = device)
    advantages = torch.zeros(B, T, device = device)
    for t in reversed(range(T)):
        gae = delta[:, t] + lam * gae * response_mask[:, t]
        advantages[:, t] = gae * response_mask[:, t]
    returns = (advantages + v_curr) * response_mask
    return advantages, returns

def train_ppo_model(
    policy_model,
    value_model,
    reference_model,
    reward_model,
    dataloader,
    epochs = 1,
    ppo_epochs = 2,
    generation_length = 160,
    beta = 0.1,
    epsilon = 0.2,
    model_dir = "../model/ppo",
    lr_policy = 1e-6,
    lr_value = 1e-6,
    warmup_steps = 100,
    max_grad_norm_policy = 10,
    max_grad_norm_value = 100,
    target_kl = 0.05,
    kl_clamp = 10
):
    global _LAST_PPO_PROGRESS_LINES
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    os.makedirs(model_dir, exist_ok = True)
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    policy_model = policy_model.to(device)
    value_model = value_model.to(device)
    reference_model = reference_model.to(device).eval()
    reward_model = reward_model.to(device).eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    for param in reward_model.parameters():
        param.requires_grad = False
    policy_optimizer = torch.optim.AdamW(policy_model.parameters(), lr = lr_policy)
    value_optimizer = torch.optim.AdamW(value_model.parameters(), lr = lr_value)
    policy_scheduler = torch.optim.lr_scheduler.LambdaLR(policy_optimizer, lr_lambda = lambda step: min(1.0, step / max(1, warmup_steps)))
    value_scheduler = torch.optim.lr_scheduler.LambdaLR(value_optimizer, lr_lambda = lambda step: min(1.0, step / max(1, warmup_steps)))
    beta = float(beta)
    beta_floor = beta
    global_step = 0

    for epoch in range(1, epochs + 1):
        _LAST_PPO_PROGRESS_LINES = 0
        total_batches = len(dataloader)
        batch_rewards = []
        batch_kls = []
        batch_entropies = []
        clipped_batches = []

        for bi, batch in enumerate(dataloader, start = 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prompt_lengths = attention_mask.sum(dim = 1)

            # Generate a full response from the current policy model
            policy_model.eval()
            with torch.no_grad():
                full_ids = policy_model.generate(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    max_new_tokens = generation_length,
                    do_sample = True,
                    temperature = 0.9,
                    pad_token_id = _TOKENIZER.eos_token_id,
                )
            full_mask = (full_ids != _TOKENIZER.pad_token_id).long()

            # Score the generated response: reward model gives a scalar quality score;
            # value model gives per-token return estimates as a baseline.
            # old_log_probs anchors the PPO ratio; ref_log_probs anchors the KL penalty.
            with torch.no_grad():
                rewards = reward_model(full_ids, full_mask)
                values = value_model(full_ids, full_mask)
                old_log_probs = log_probs(policy_model, full_ids, full_mask, prompt_lengths)
                ref_log_probs = log_probs(reference_model, full_ids, full_mask, prompt_lengths)

            # response_mask selects generated (non-prompt, non-pad) token positions.
            # log_ratio = log(π_old / π_ref): how much more or less likely the current policy
            # is to produce each token compared to the reference model.
            positions = torch.arange(1, full_ids.shape[1], device = device).unsqueeze(0)
            response_mask = (full_mask[:, 1:].float() * (positions >= prompt_lengths.unsqueeze(1)).float())
            log_ratio = (old_log_probs - ref_log_probs) * response_mask
            response_lengths = response_mask.sum(dim = 1)
            n_response_tokens = response_mask.sum()

            # k3 KL estimator: exp(-d) + d - 1 where d = log(π_old / π_ref).
            # Unbiased and always ≥ 0, prevents runaway collapse when KL goes negative.
            # kl_mean: per-token KL averaged over the batch, used by the beta controller.
            # drift: mean log-ratio that quantifies how much policy has moved away from the reference.
            kl_sum = ((torch.exp(-log_ratio) + log_ratio - 1).clamp(max = kl_clamp) * response_mask).sum(dim = 1)
            kl_mean = (kl_sum / (response_lengths + 1e-8)).mean().item()
            drift = float(((ref_log_probs - old_log_probs) * response_mask).sum().item() / (response_mask.sum().item() + 1e-8))

            # Returns are the terminal reward redistributed per-token via GAE.
            # Advantages measure how much each token's return exceeds the value model baseline.
            # Advantages are normalized to unit variance for training consistency.
            # Entropy tracks generation health (GPT-2 reference ranges):
            #   < 1: collapsed | 2–3: healthy fine-tuned | 3–5: general LM | > 10: noise
            advantages, returns = compute_gae(values.detach(), rewards.detach(), response_mask)
            adv_mean = (advantages * response_mask).sum() / (n_response_tokens + 1e-8)
            adv_var = ((advantages - adv_mean).pow(2) * response_mask).sum() / (n_response_tokens + 1e-8)
            advantages = (advantages - adv_mean) / (adv_var.sqrt() + 1e-8) * response_mask
            batch_entropy = float((-old_log_probs * response_mask).sum().item() / (n_response_tokens.item() + 1e-8))

            for ppo_step in range(ppo_epochs):
                # Recompute log probs with gradient; PPO clipped objective + KL penalty.
                new_log_probs = log_probs(policy_model, full_ids, full_mask, prompt_lengths)
                ratio = torch.exp(new_log_probs - old_log_probs.detach())
                ratio_clamped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
                token_loss = -torch.min(ratio * advantages.detach(), ratio_clamped * advantages.detach())
                ppo_loss = (token_loss * response_mask).sum() / (n_response_tokens + 1e-8)
                new_log_ratio = ((new_log_probs - ref_log_probs.detach()) * response_mask)
                kl_div = ((torch.exp(-new_log_ratio.clamp(min = -20)) + new_log_ratio - 1) * response_mask).sum() / (n_response_tokens + 1e-8)
                policy_loss = ppo_loss + beta * kl_div

                # Policy optimiser step and gradient clipping
                policy_optimizer.zero_grad()
                policy_loss.backward()
                ppo_grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_grad_norm_policy if global_step > warmup_steps else float("inf"))
                ppo_grad_norm_value = float(ppo_grad_norm.item()) if hasattr(ppo_grad_norm, "item") else float(ppo_grad_norm)
                if not math.isfinite(ppo_grad_norm_value):
                    ensure_newline()
                    print(f"policy grad non-finite skipped: epoch {epoch}, batch {bi}, step {ppo_step + 1}: norm = {ppo_grad_norm_value}")
                    policy_optimizer.zero_grad()
                elif global_step > warmup_steps and ppo_grad_norm_value > max_grad_norm_policy:
                    ensure_newline()
                    clipped_batches.append({"epoch": epoch, "batch_index": bi, "ppo_step": ppo_step + 1, "type": "policy", "grad_norm": ppo_grad_norm_value})
                    print(f"policy grad clipped: epoch {epoch}, batch {bi}, step {ppo_step + 1}: norm = {ppo_grad_norm_value:.6g}")
                    policy_optimizer.step()
                else:
                    policy_optimizer.step()
                policy_scheduler.step()

                # Value loss is MSE between updated value estimates and GAE returns.
                new_values = value_model(full_ids, full_mask)
                value_loss = ((new_values[:, :-1] - returns.detach()).pow(2) * response_mask).sum() / (n_response_tokens + 1e-8)

                # Value optimiser step and gradient clipping
                value_optimizer.zero_grad()
                value_loss.backward()
                value_grad_norm = torch.nn.utils.clip_grad_norm_(value_model.parameters(), max_grad_norm_value)
                val_grad_norm_value = float(value_grad_norm.item()) if hasattr(value_grad_norm, "item") else float(value_grad_norm)
                if not math.isfinite(val_grad_norm_value):
                    ensure_newline()
                    print(f"value grad non-finite skipped: epoch {epoch}, batch {bi}, step {ppo_step + 1}: norm = {val_grad_norm_value}")
                    value_optimizer.zero_grad()
                elif val_grad_norm_value > max_grad_norm_value:
                    ensure_newline()
                    clipped_batches.append({"epoch": epoch, "batch_index": bi, "ppo_step": ppo_step + 1, "type": "value", "grad_norm": val_grad_norm_value})
                    print(f"value grad clipped: epoch {epoch}, batch {bi}, step {ppo_step + 1}: norm = {val_grad_norm_value:.6g}")
                    value_optimizer.step()
                else:
                    value_optimizer.step()
                value_scheduler.step()

                global_step += 1
                show_progress_ppo(
                    bi, total_batches, epoch, ppo_step, ppo_epochs,
                    reward = float(rewards.mean().item()),
                    kl = kl_mean,
                    beta = beta,
                    drift = drift,
                    entropy = batch_entropy,
                    grad_norm_policy = ppo_grad_norm_value,
                    grad_norm_value = val_grad_norm_value,
                )
            
            # Adaptive beta: raise if KL too high, lower if too low
            if kl_mean > 1.5 * target_kl:
                beta = min(2, beta * 1.3)
            elif 0 < kl_mean < 0.5 * target_kl:
                beta = max(beta_floor, beta / 1.02)

            batch_rewards.append(float(rewards.mean().item()))
            batch_kls.append(kl_mean)
            batch_entropies.append(batch_entropy)

        ensure_newline()
        history = {"rewards": batch_rewards, "kl": batch_kls, "entropy": batch_entropies}
        tag = f"e{epoch}"
        epoch_dir = os.path.join(model_dir, f"{run_id}-{tag}")
        os.makedirs(epoch_dir, exist_ok = True)
        torch.save(policy_model, os.path.join(epoch_dir, f"policy-{run_id}-{tag}.pt"))
        torch.save(value_model, os.path.join(epoch_dir, f"value-{run_id}-{tag}.pt"))
        torch.save(history, os.path.join(epoch_dir, f"history-{run_id}-{tag}.pt"))
        with open(os.path.join(epoch_dir, f"bad_batches-{run_id}-{tag}.json"), "w", encoding = "utf-8") as f:
            json.dump(clipped_batches, f, ensure_ascii = False, indent = 2)

    return policy_model, value_model
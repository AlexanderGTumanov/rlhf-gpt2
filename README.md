# RLHF Fine-Tuning of GPT-2

Reinforcement Learning from Human Feedback (RLHF) aims at improving language model performance based on collected human feedback. Getting a language model to follow human preferences is harder than it looks. The training data is nothing more than binary judgements over a series of prompt-response pairs, which carries almost no direct learning signal, and models reliably find ways to game it.

Proximal Policy Optimization (PPO) addresses this with a four-model architecture. A *reward model* is trained first to score responses from human preference data. A *reference model* is produced by supervised fine-tuning but without incorporating any of the feedback. It serves as a baseline for the *policy model*, which is trained in the PPO loop to maximize the reward model's score. A fourth model, the *value model*, is trained alongside it to dynamically mimic the reward model's scores. It provides a cleaner training signal for the policy model that reflects genuine human preference rather than incidental factors like the base complexity of the prompt. Another core feature of the PPO loop is the Kullback-Leibler (KL) penalty, which prevents the policy model from drifting too far from the reference and collapsing.

This project builds the full pipeline from scratch, with GPT-2 serving as the backbone for all models, without relying on publicly available implementations such as TRL. The models are trained on the Anthropic HH-RLHF dataset on a 32 GB Apple M1. The policy achieves a ~70% improvement in reward over the reference baseline at ~17σ significance. Trained models are available on Hugging Face:

[https://huggingface.co/AlexanderGTumanov/rlhf-gpt2](https://huggingface.co/AlexanderGTumanov/rlhf-gpt2)

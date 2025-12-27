---
layout: post
title: "A GSM8K Study Comparing Pseudo Labeling for SFT and GRPO"
date: 2025-12-26
description: "Pseudo-labeling doesn't help when you have GRPO. GRPO on 1K examples matches SFT on 5K."
---

We tested whether consensus-based pseudo-labeling can expand training data for math reasoning. The setup: train on 1K labeled GSM8K problems, pseudo-label 4K more using majority vote over K=8 samples, train on the combined set. Result: no improvement over the 1K baseline. But switching from SFT to GRPO on the same 1K labeled examples matched what SFT needed 5K real labels to achieve.

This project was motivated by a research direction from [Thinking Machines Lab](https://thinkingmachines.ai/blog/call-for-community-projects/): adapting noisy student self-distillation to leverage unlabeled data for LLM training.

## Pseudo-labeling

For each unlabeled problem, generate K=8 completions at temperature 0.7 and extract the final numerical answer from each. The majority answer becomes the pseudo-label. If 6/8 completions produce "42", the pseudo-label is 42 with 75% confidence. This achieved 52% accuracy against ground truth - roughly half the pseudo-labels were wrong.

## SFT Experiments

Model: Qwen-2.5-0.5B-Instruct. Training: standard supervised fine-tuning on problem-solution pairs. Evaluation: 500 test examples, same seed across all conditions.

| Condition | Test Accuracy |
|-----------|---------------|
| Untrained | 21.0% |
| 4K pseudo-labeled only | 23.6% |
| 1K labeled (baseline) | 27.0% |
| 1K labeled + 4K pseudo | 26.8% |
| 5K real labels (oracle) | 32.4% |

The noisy student condition matched the baseline exactly - 4K additional pseudo-labeled examples contributed nothing. Note that noisy student used 5x more compute (2500 steps vs 500 steps) yet showed no improvement.

We tried filtering to only high-confidence pseudo-labels (>=75% agreement). This yielded 981 examples at 93.7% accuracy - still no improvement over baseline (27.6% vs 27.0%). Why high-quality pseudo-labels didn't help is unclear: possibly the quantity was insufficient, possibly the remaining 6% errors still mattered, possibly the examples don't add signal the model can use. We suspect the high-confidence examples were problems the model already solved correctly, providing no new learning signal.

The oracle result (32.4%) confirms the model has capacity to learn more - it just can't learn from pseudo-labeled data.

## GRPO Experiments

We replaced SFT with GRPO (Group Relative Policy Optimization), keeping all data splits identical. The reward function: 1 if extracted answer matches ground truth, 0 otherwise.

| Condition | Test Accuracy | vs SFT |
|-----------|---------------|--------|
| GRPO on 1K labeled | 32.6% | +5.6pp over SFT baseline |
| GRPO on 1K + 4K pseudo | 30.6% | +3.8pp over SFT noisy student |

GRPO on 1K labeled examples matched SFT oracle performance (32.6% vs 32.4%). Same data, different training method, 5x sample efficiency improvement.

GRPO with noisy pseudo-labels (30.6%) beat SFT with noisy pseudo-labels (26.8%) by 3.8pp - confirming GRPO is more robust to label noise. But it still underperformed GRPO baseline by 2pp. Noisy labels hurt both methods; GRPO just handles them better.

## Why GRPO Outperforms SFT

These results point to a fundamental difference in the learning signals provided by the two training paradigms.
**For reasoning tasks, the primary learning signal is outcome correctness, not token-level imitation.**

SFT minimizes cross-entropy loss over the full solution text. This penalizes the model whenever its reasoning diverges from the reference trajectory—even if it reaches the correct answer via a valid alternative path, uses different intermediate variables, or orders steps differently.

GRPO, by contrast, optimizes a binary reward based solely on answer correctness. The model is free to use whatever internal reasoning strategy proves effective, reinforcing behaviors that lead to correct outcomes without constraining the form of the solution. Given that the model already possesses substantial mathematical knowledge from pretraining, GRPO serves to identify and amplify successful reasoning strategies rather than enforce imitation of a single trace.

## Getting GRPO to Work

The default GRPO configuration didn't work out of the box. Key issues and fixes:

**Zero loss**: With `num_generations=4`, all samples in a group often received the same reward (all 0 or all 1). Zero variance means zero advantage, which means zero gradient. Fix: increase to `num_generations=8`.

**Degenerate outputs**: Model would produce the correct answer, then continue generating repetitive patterns like "####. ####. ####." indefinitely. Caused by `beta=0.0` (no KL penalty against the reference policy). Fix: set `beta=0.04` and reduce temperature from 0.7 to 0.3.

**Loss type**: The default `dapo` loss type produced zero loss even with variance. Fix: switch to `dr_grpo` which uses a constant denominator.

Final working configuration:
```
temperature=0.3
num_generations=8
loss_type=dr_grpo
learning_rate=1e-5
beta=0.04
max_completion_length=256
```

## Results Summary

| Method | 1K labeled | 1K + 4K pseudo | 5K real |
|--------|------------|----------------|---------|
| SFT | 27.0% | 26.8% | 32.4% |
| GRPO | 32.6% | 30.6% | - |

1. GRPO on 1K labeled matches SFT on 5K real labels
2. Pseudo-labels don't help SFT at all
3. Pseudo-labels hurt GRPO by 2pp, but GRPO still beats all SFT conditions
4. GRPO is more robust to label noise than SFT (+3.8pp on noisy data)

## Conclusions

The noisy student approach doesn't help when you have GRPO. GRPO extracts more signal from 1K clean examples than SFT can from 5K - the pseudo-labeling machinery becomes unnecessary overhead that slightly degrades performance.

This doesn't mean pseudo-labeling is useless for LLMs generally. Our pseudo-labels were only 46-52% accurate. With a stronger teacher model or higher K, accuracy could improve enough to help. We also only tested one iteration - iterative refinement might compound gains.

## Limitations

- Single model size (0.5B parameters)
- Single dataset (GSM8K)
- Single pseudo-labeling approach (majority vote, K=8)
- No iterative refinement
- Statistical significance is marginal for some comparisons (n=500, 95% CI ~4.5pp)

Code: [github.com/omarkhursheed/noisy-student](https://github.com/omarkhursheed/noisy-student)

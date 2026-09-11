# Fine-Tuning
Learn how to finetune the llm model
# Gemma 3 1B QLoRA Fine-Tuning

A parameter-efficient fine-tuning project for adapting Google's Gemma 3 1B instruction-tuned language model using **QLoRA**, **4-bit NF4 quantization**, and **Supervised Fine-Tuning (SFT)** on the Databricks Dolly 15K instruction-following dataset.

## Overview

This project demonstrates how to fine-tune a large language model while keeping the pretrained model weights frozen and training only a small set of additional LoRA parameters.

The workflow combines:

- 4-bit NF4 quantization for memory-efficient model loading
- LoRA for parameter-efficient adaptation
- Supervised Fine-Tuning (SFT)
- Hugging Face Transformers
- Hugging Face PEFT
- TRL
- Databricks Dolly 15K dataset

## Architecture

```text
                    Databricks Dolly 15K
                            │
                            ▼
                    Data Preparation
                            │
                            ▼
                       Tokenization
                            │
                            ▼
                 ┌─────────────────────┐
                 │     Gemma 3 1B      │
                 │                     │
                 │  Pretrained Weights │
                 │       🔒 Frozen     │
                 └──────────┬──────────┘
                            │
                    + LoRA Adapters
                       A + B Matrices
                         🎯 Trainable
                            │
                            ▼
                   Supervised Fine-Tuning
                            │
                            ▼
                  Fine-Tuned LoRA Adapter

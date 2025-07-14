#!/usr/bin/env python3

from datasets import load_dataset
from utils import display_dataset

print("Loading dataset...")
train_dataset = load_dataset("banghua/DL-SFT-Dataset")["train"]

print(f"Dataset size: {len(train_dataset)} examples")
print("\nFirst 3 examples:")
display_dataset(train_dataset) 
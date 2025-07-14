#!/usr/bin/env python3

from utils import load_model_and_tokenizer, test_model_with_questions

# Test questions
USE_GPU = False
questions = [
    "Give me an 1-sentence introduction of LLM.",
    "Calculate 1+1-1",
    "What's the difference between thread and process?"
]

print("Loading base model...")
model, tokenizer = load_model_and_tokenizer("Qwen/Qwen3-0.6B-Base", USE_GPU)
test_model_with_questions(model, tokenizer, questions, 
                          title="Base Model (Before SFT) Output")
del model, tokenizer
print("\nBase model testing completed!") 
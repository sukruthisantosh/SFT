# Helper functions for model loading, testing, and dataset display

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

def load_model_and_tokenizer(model_name, use_gpu):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if use_gpu:
        model = model.to('cuda')
    return model, tokenizer

def generate_responses(model, tokenizer, input_texts):
    inputs = tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True)
    if next(model.parameters()).is_cuda:
        inputs = {key: value.to('cuda') for key, value in inputs.items()}
    outputs = model.generate(**inputs)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def test_model_with_questions(model, tokenizer, questions, title="Model Output"):
    print(f"=== {title} ===")
    responses = generate_responses(model, tokenizer, questions)
    for question, response in zip(questions, responses):
        print(f"Q: {question}\nA: {response}\n")

def display_dataset(dataset):
    df = pd.DataFrame(dataset)
    print(df.head())  # Display the first few rows of the dataset
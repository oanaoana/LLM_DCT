"""
Shared fixtures for tests
"""
import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="session")
def tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

@pytest.fixture(scope="session")
def original_model(device):
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()
    return model

@pytest.fixture
def test_prompts():
    return [
        "The quick brown fox",
        "Hello, my name is",
        "Once upon a time"
    ]
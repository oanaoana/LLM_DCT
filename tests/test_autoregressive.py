"""
Tests for autoregressive generation with patched attention
"""
import torch
import pytest
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from models.sparsification import apply_attention_sparsification
import transforms.dct as dct


@pytest.fixture
def models_and_tokenizer():
    """Create original and patched models for testing"""
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    device = torch.device("cpu")  # Use CPU for testing

    # Original model
    original_model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()

    # Identity function for no compression
    identity_fn = lambda x: x

    # Patched model with identity functions
    identity_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    identity_model = apply_attention_sparsification(
        identity_model,
        sparsify_q_fn=identity_fn,
        sparsify_k_fn=identity_fn,
        sparsify_v_fn=identity_fn
    ).eval()

    # DCT compressed model
    hidden_size = original_model.config.hidden_size
    compressed_dim = int(hidden_size * 0.5)  # 50% compression
    basis, inverse = dct.build_dct_basis(hidden_size, compressed_dim)

    compressed_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    compressed_model = apply_attention_sparsification(
        compressed_model,
        sparsify_q_fn=lambda x: dct.dct_project(x, basis),
        sparsify_k_fn=lambda x: dct.dct_project(x, basis),
        sparsify_v_fn=lambda x: dct.dct_project(x, basis),
        backproject_fn=lambda x: dct.dct_backproject(x, inverse),
        compressed_dim=compressed_dim
    ).eval()

    return {
        "original": original_model,
        "identity": identity_model,
        "compressed": compressed_model,
        "tokenizer": tokenizer,
        "device": device
    }


@pytest.mark.parametrize("prompt", [
    "The quick brown fox",
    "Hello, my name is",
    "Once upon a time"
])
def test_identity_generation_matches_original(models_and_tokenizer, prompt):
    """Test that identity-patched model matches original model output"""
    original_model = models_and_tokenizer["original"]
    identity_model = models_and_tokenizer["identity"]
    tokenizer = models_and_tokenizer["tokenizer"]
    device = models_and_tokenizer["device"]

    # Set the same seed for both
    torch.manual_seed(42)
    original_inputs = tokenizer(prompt, return_tensors="pt").to(device)
    original_outputs = original_model.generate(
        **original_inputs,
        max_new_tokens=20,
        do_sample=False  # Greedy decoding for deterministic results
    )
    original_text = tokenizer.decode(original_outputs[0], skip_special_tokens=True)

    torch.manual_seed(42)
    identity_inputs = tokenizer(prompt, return_tensors="pt").to(device)
    identity_outputs = identity_model.generate(
        **identity_inputs,
        max_new_tokens=20,
        do_sample=False  # Greedy decoding for deterministic results
    )
    identity_text = tokenizer.decode(identity_outputs[0], skip_special_tokens=True)

    # The generations should match exactly
    assert original_text == identity_text, f"Identity model output differs from original:\n{original_text}\nvs\n{identity_text}"


@pytest.mark.parametrize("prompt", [
    "The quick brown fox",
    "Hello, my name is",
    "Once upon a time"
])
def test_compressed_generation_quality(models_and_tokenizer, prompt):
    """Test that compressed model output is similar to original (not exact match)"""
    original_model = models_and_tokenizer["original"]
    compressed_model = models_and_tokenizer["compressed"]
    tokenizer = models_and_tokenizer["tokenizer"]
    device = models_and_tokenizer["device"]

    # Set the same seed for both
    torch.manual_seed(42)
    original_inputs = tokenizer(prompt, return_tensors="pt").to(device)
    original_outputs = original_model.generate(
        **original_inputs,
        max_new_tokens=20,
        do_sample=False
    )
    original_text = tokenizer.decode(original_outputs[0], skip_special_tokens=True)

    torch.manual_seed(42)
    compressed_inputs = tokenizer(prompt, return_tensors="pt").to(device)
    compressed_outputs = compressed_model.generate(
        **compressed_inputs,
        max_new_tokens=20,
        do_sample=False
    )
    compressed_text = tokenizer.decode(compressed_outputs[0], skip_special_tokens=True)

    # Print outputs for inspection
    print(f"\nPrompt: '{prompt}'")
    print(f"Original: '{original_text}'")
    print(f"Compressed: '{compressed_text}'")

    # Check that outputs start with the prompt
    assert original_text.startswith(prompt), "Original output doesn't start with prompt"
    assert compressed_text.startswith(prompt), "Compressed output doesn't start with prompt"

    # For compressed model, we expect similar but not necessarily identical output
    # Check that at least some tokens match
    original_tokens = tokenizer.tokenize(original_text)
    compressed_tokens = tokenizer.tokenize(compressed_text)

    # Calculate token overlap percentage
    min_len = min(len(original_tokens), len(compressed_tokens))
    matching = sum(1 for i in range(min_len) if original_tokens[i] == compressed_tokens[i])
    match_percentage = matching / min_len

    # We expect some reasonable overlap, but not necessarily 100%
    # This threshold might need adjustment based on your compression method
    assert match_percentage > 0.5, f"Token match too low: {match_percentage:.2f}"

    return match_percentage


@pytest.mark.parametrize("model_type", ["identity", "compressed"])
def test_autoregressive_single_token_generation(models_and_tokenizer, model_type):
    """
    Test token-by-token autoregressive generation to detect any
    accumulation of errors during the autoregressive process
    """
    original_model = models_and_tokenizer["original"]
    test_model = models_and_tokenizer[model_type]
    tokenizer = models_and_tokenizer["tokenizer"]
    device = models_and_tokenizer["device"]

    prompt = "The quick brown fox jumps over"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate each token one-by-one (manually doing autoregressive generation)
    prompt_length = inputs.input_ids.shape[1]
    max_new_tokens = 10

    # Original model generation
    torch.manual_seed(42)
    with torch.no_grad():
        original_outputs = original_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
    original_tokens = original_outputs[0][prompt_length:]  # Get only the new tokens

    # Test model token-by-token generation
    torch.manual_seed(42)
    test_inputs = inputs.input_ids.clone()
    generated_tokens = []

    for _ in range(max_new_tokens):
        with torch.no_grad():
            token_inputs = {"input_ids": test_inputs}
            outputs = test_model(**token_inputs)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            # Add token to generated list
            generated_tokens.append(next_token.item())

            # Add to inputs for next iteration
            test_inputs = torch.cat([test_inputs, next_token], dim=1)

    # Compare tokens
    manual_generation = torch.tensor(generated_tokens)

    # Print debug info
    print(f"\n{model_type.capitalize()} Model Token-by-Token Generation:")
    print(f"Original: {tokenizer.decode(original_tokens)}")
    print(f"Manual: {tokenizer.decode(manual_generation)}")

    if model_type == "identity":
        # Identity model should match original exactly
        assert torch.all(original_tokens == manual_generation), "Identity model token-by-token generation differs from original"
    else:
        # Compressed model may differ, but check for some similarity
        matching = sum(1 for i in range(len(manual_generation)) if manual_generation[i] == original_tokens[i])
        match_percentage = matching / len(manual_generation)
        assert match_percentage > 0.3, f"Token match too low: {match_percentage:.2f}"
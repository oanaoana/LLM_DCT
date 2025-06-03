"""
End-to-end tests for compressed models
"""
import torch
import pytest
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from models.sparsification import apply_attention_sparsification
from evaluation.metrics import calculate_bleu
import transforms.dct as dct
import transforms.pswf as pswf


@pytest.fixture(scope="module")
def models_and_tokenizer():
    """Create models for testing"""
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    device = torch.device("cpu")  # Use CPU for testing

    # Original model
    original_model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()
    hidden_size = original_model.config.hidden_size

    # Create models with different compression methods
    models = {
        "original": original_model,
        "tokenizer": tokenizer,
        "device": device
    }

    # DCT model
    compressed_dim = int(hidden_size * 0.5)
    basis, inverse = dct.build_dct_basis(hidden_size, compressed_dim)

    dct_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    dct_model = apply_attention_sparsification(
        dct_model,
        sparsify_q_fn=lambda x: dct.dct_project(x, basis),
        sparsify_k_fn=lambda x: dct.dct_project(x, basis),
        sparsify_v_fn=lambda x: dct.dct_project(x, basis),
        backproject_fn=lambda x: dct.dct_backproject(x, inverse),
        compressed_dim=compressed_dim
    ).eval()
    models["dct"] = dct_model

    # PSWF model
    pswf_basis, pswf_inverse = pswf.build_pswf_basis(hidden_size, compressed_dim)

    pswf_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    pswf_model = apply_attention_sparsification(
        pswf_model,
        sparsify_q_fn=lambda x: pswf.pswf_project(x, pswf_basis),
        sparsify_k_fn=lambda x: pswf.pswf_project(x, pswf_basis),
        sparsify_v_fn=lambda x: pswf.pswf_project(x, pswf_basis),
        backproject_fn=lambda x: pswf.pswf_backproject(x, pswf_inverse),
        compressed_dim=compressed_dim
    ).eval()
    models["pswf"] = pswf_model

    # Identity model (patched but no compression)
    identity_fn = lambda x: x

    identity_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    identity_model = apply_attention_sparsification(
        identity_model,
        sparsify_q_fn=identity_fn,
        sparsify_k_fn=identity_fn,
        sparsify_v_fn=identity_fn,
        backproject_fn=None,
    ).eval()
    models["identity"] = identity_model

    return models


def test_identity_model_equivalence(models_and_tokenizer):
    """Test that identity patched model gives identical outputs to original model"""
    original_model = models_and_tokenizer["original"]
    identity_model = models_and_tokenizer["identity"]
    tokenizer = models_and_tokenizer["tokenizer"]
    device = models_and_tokenizer["device"]

    prompts = [
        "The quick brown fox jumps over",
        "In a world where everything is",
        "Once upon a time in a galaxy"
    ]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Set fixed random seed for deterministic generation
        torch.manual_seed(42)
        with torch.no_grad():
            orig_output = original_model.generate(**inputs, max_new_tokens=20, do_sample=False)

        torch.manual_seed(42)
        with torch.no_grad():
            identity_output = identity_model.generate(**inputs, max_new_tokens=20, do_sample=False)

        # Check that outputs are identical
        assert torch.equal(orig_output, identity_output), "Identity model outputs differ from original"

        # Check decoded text
        orig_text = tokenizer.decode(orig_output[0], skip_special_tokens=True)
        identity_text = tokenizer.decode(identity_output[0], skip_special_tokens=True)
        assert orig_text == identity_text, "Identity model text differs from original"


@pytest.mark.parametrize("model_type", ["dct", "pswf"])
def test_compressed_model_quality(models_and_tokenizer, model_type):
    """Test that compressed models maintain reasonable quality"""
    original_model = models_and_tokenizer["original"]
    compressed_model = models_and_tokenizer[model_type]
    tokenizer = models_and_tokenizer["tokenizer"]
    device = models_and_tokenizer["device"]

    prompts = [
        "The meaning of life is",
        "Once upon a time there was",
        "The best way to learn programming is"
    ]

    # Collect BLEU scores
    bleu1_scores = []
    bleu4_scores = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate with fixed seed
        torch.manual_seed(42)
        with torch.no_grad():
            orig_output = original_model.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=0.7)

        torch.manual_seed(42)
        with torch.no_grad():
            comp_output = compressed_model.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=0.7)

        # Decode texts
        orig_text = tokenizer.decode(orig_output[0], skip_special_tokens=True)
        comp_text = tokenizer.decode(comp_output[0], skip_special_tokens=True)

        # Extract only the generated portion (after the prompt)
        prompt_len = len(prompt)
        orig_generated = orig_text[prompt_len:].strip()
        comp_generated = comp_text[prompt_len:].strip()

        # Calculate BLEU scores
        bleu_scores = calculate_bleu(orig_generated, comp_generated)
        bleu1_scores.append(bleu_scores["bleu1"])
        bleu4_scores.append(bleu_scores["bleu4"])

    # Calculate average scores
    avg_bleu1 = sum(bleu1_scores) / len(bleu1_scores)
    avg_bleu4 = sum(bleu4_scores) / len(bleu4_scores)

    # Check quality thresholds based on compression type
    if model_type == "dct":
        min_bleu1 = 0.3  # Minimum expected BLEU-1 score for DCT
    else:  # PSWF
        min_bleu1 = 0.25  # PSWF might have slightly lower quality

    # Assert quality thresholds
    assert avg_bleu1 > min_bleu1, f"{model_type} model BLEU-1 score too low: {avg_bleu1:.4f}"

    # Log results
    print(f"\n{model_type.upper()} compression quality:")
    print(f"Average BLEU-1: {avg_bleu1:.4f}, Average BLEU-4: {avg_bleu4:.4f}")


def test_model_inference_shapes(models_and_tokenizer):
    """Test that all models produce correct output shapes"""
    tokenizer = models_and_tokenizer["tokenizer"]
    device = models_and_tokenizer["device"]

    prompt = "Testing output shapes"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Test each model type
    for model_name, model in models_and_tokenizer.items():
        if model_name in ["tokenizer", "device"]:
            continue

        with torch.no_grad():
            # Forward pass
            outputs = model(**inputs)

            # Check logits shape
            batch_size, seq_len, vocab_size = outputs.logits.shape
            assert batch_size == 1, f"{model_name} model: Wrong batch size"
            assert seq_len == inputs.input_ids.shape[1], f"{model_name} model: Wrong sequence length"
            assert vocab_size == model.config.vocab_size, f"{model_name} model: Wrong vocabulary size"

            # Generate tokens
            gen_outputs = model.generate(**inputs, max_new_tokens=5)

            # Check generation shape
            assert gen_outputs.shape[0] == 1, f"{model_name} model: Wrong generation batch size"
            assert gen_outputs.shape[1] > inputs.input_ids.shape[1], f"{model_name} model: No tokens generated"


def test_model_gradients(models_and_tokenizer):
    """Test that compressed models can be trained (gradients flow properly)"""
    tokenizer = models_and_tokenizer["tokenizer"]
    device = models_and_tokenizer["device"]

    prompt = "Checking gradient flow"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Add labels for loss calculation (shift inputs right)
    labels = inputs.input_ids.clone()

    # Test each model type
    for model_name, model in models_and_tokenizer.items():
        if model_name in ["tokenizer", "device", "original"]:
            continue

        # Ensure training mode
        model.train()

        # Forward pass with labels
        outputs = model(**inputs, labels=labels)

        # Check that loss is calculated
        assert outputs.loss is not None, f"{model_name} model: No loss calculated"
        assert outputs.loss.item() > 0, f"{model_name} model: Loss value suspicious"

        # Backpropagate
        outputs.loss.backward()

        # Check that gradients exist
        has_grads = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grads = True
                break

        assert has_grads, f"{model_name} model: No gradients found"

        # Reset model to eval mode
        model.eval()
        model.zero_grad()


if __name__ == "__main__":
    # For manual testing
    models = models_and_tokenizer()
    test_identity_model_equivalence(models)
    test_compressed_model_quality(models, "dct")
    test_compressed_model_quality(models, "pswf")
    test_model_inference_shapes(models)
    test_model_gradients(models)
    print("All tests passed!")
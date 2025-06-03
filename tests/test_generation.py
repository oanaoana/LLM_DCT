"""
Tests for autoregressive text generation to verify patched model behaves correctly
"""
import torch
import pytest
from transformers import GPT2LMHeadModel
from models.sparsification import apply_attention_sparsification
from utils.generation import generate_text

@pytest.mark.parametrize("sparsify_fn", [
    lambda x: x,  # Identity function
    lambda x: x[:, :x.size(1)//2],  # Half dimension (with truncation)
])
def test_identity_generation(original_model, tokenizer, test_prompts, device, sparsify_fn):
    """Test if generation with identity function matches the original model"""
    # Create patched model with provided sparsification function
    patched_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    patched_model = apply_attention_sparsification(
        patched_model,
        sparsify_q_fn=sparsify_fn,
        sparsify_k_fn=sparsify_fn,
        sparsify_v_fn=sparsify_fn,
        # If truncation, add backprojection
        backproject_fn=lambda x: torch.cat([x, torch.zeros(x.size(0),
                                                         original_model.config.hidden_size-x.size(1),
                                                         device=x.device)], dim=1)
                              if sparsify_fn.__name__ != "<lambda>" else None
    ).eval()

    for prompt in test_prompts:
        # Set same seed for both
        torch.manual_seed(42)
        original_text = generate_text(
            original_model, tokenizer, prompt, max_new_tokens=15, device=device
        )

        torch.manual_seed(42)
        patched_text = generate_text(
            patched_model, tokenizer, prompt, max_new_tokens=15, device=device
        )

        # Compare token by token
        assert original_text == patched_text, f"Generated texts differ: '{original_text}' vs '{patched_text}'"


@pytest.mark.parametrize("compression_ratio", [0.75, 0.5, 0.25])
@pytest.mark.parametrize("transform_type", ["dct", "pswf"])
def test_transform_generation_quality(original_model, tokenizer, test_prompts,
                                    device, compression_ratio, transform_type):
    """Test generation quality with different compression methods"""
    import transforms.dct as dct
    import transforms.pswf as pswf

    # Create compressed model
    compressed_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    # Get model dimensions
    hidden_size = compressed_model.config.hidden_size
    num_heads = compressed_model.config.n_head

    # Calculate compressed dimension
    compressed_dim = int(hidden_size * compression_ratio)
    compressed_dim = (compressed_dim // num_heads) * num_heads

    # Build transform basis
    if transform_type == "dct":
        basis, inverse = dct.build_dct_basis(hidden_size, compressed_dim)
        project_fn = lambda x: dct.dct_project(x, basis)
        backproject_fn = lambda x: dct.dct_backproject(x, inverse)
    else:  # pswf
        basis, inverse = pswf.build_pswf_basis(hidden_size, compressed_dim)
        project_fn = lambda x: pswf.pswf_project(x, basis)
        backproject_fn = lambda x: pswf.pswf_backproject(x, inverse)

    # Apply compression to the model
    compressed_model = apply_attention_sparsification(
        compressed_model,
        sparsify_q_fn=project_fn,
        sparsify_k_fn=project_fn,
        sparsify_v_fn=project_fn,
        backproject_fn=backproject_fn,
        compressed_dim=compressed_dim
    ).eval()

    from evaluation.metrics import calculate_bleu

    bleu_scores = []

    # Test each prompt
    for prompt in test_prompts:
        # Generate with fixed seed
        torch.manual_seed(0)
        original_text = generate_text(
            original_model, tokenizer, prompt, max_new_tokens=30, device=device
        )

        torch.manual_seed(0)
        compressed_text = generate_text(
            compressed_model, tokenizer, prompt, max_new_tokens=30, device=device
        )

        # Calculate BLEU score
        # Only compare the generated portions (exclude the prompt)
        original_generated = original_text[len(prompt):].strip()
        compressed_generated = compressed_text[len(prompt):].strip()

        bleu = calculate_bleu(original_generated, compressed_generated)
        bleu_scores.append(bleu['bleu1'])

    # Average BLEU-1 score should be above a threshold
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    # Different expected thresholds based on compression
    expected_threshold = 0.8 if compression_ratio >= 0.5 else 0.5

    assert avg_bleu >= expected_threshold, f"BLEU-1 score too low: {avg_bleu:.4f} < {expected_threshold}"
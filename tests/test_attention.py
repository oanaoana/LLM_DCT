"""
Tests for patched attention mechanism
"""
import torch
import pytest
from transformers import GPT2LMHeadModel
from models.sparsification import apply_attention_sparsification
from models.attention_patch import PatchedCausalSelfAttention
import transforms.dct as dct


@pytest.fixture
def attention_models():
    """Create models with original and patched attention"""
    model_name = "gpt2"
    device = torch.device("cpu")

    # Original model
    original_model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()

    # Extract original attention from first block
    original_attn = original_model.transformer.h[0].attn

    # Create identity patched attention
    identity_attn = PatchedCausalSelfAttention(
        original_attn,
        sparsify_q_fn=lambda x: x,
        sparsify_k_fn=lambda x: x,
        sparsify_v_fn=lambda x: x
    )

    # Create compressed attention
    hidden_size = original_model.config.hidden_size
    compressed_dim = int(hidden_size * 0.5)
    basis, inverse = dct.build_dct_basis(hidden_size, compressed_dim)

    compressed_attn = PatchedCausalSelfAttention(
        original_attn,
        sparsify_q_fn=lambda x: dct.dct_project(x, basis),
        sparsify_k_fn=lambda x: dct.dct_project(x, basis),
        sparsify_v_fn=lambda x: dct.dct_project(x, basis),
        backproject_fn=lambda x: dct.dct_backproject(x, inverse),
        compressed_dim=compressed_dim
    )

    return {
        "original": original_attn,
        "identity": identity_attn,
        "compressed": compressed_attn,
        "hidden_size": hidden_size,
        "compressed_dim": compressed_dim
    }


def test_identity_attention_matches_original(attention_models):
    """Test if identity-patched attention matches original attention"""
    original_attn = attention_models["original"]
    identity_attn = attention_models["identity"]

    # Create random input
    batch_size = 2
    sequence_length = 10
    hidden_size = attention_models["hidden_size"]

    # Same input for both
    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, sequence_length, hidden_size)

    # Get outputs
    with torch.no_grad():
        original_output = original_attn(input_tensor)
        identity_output = identity_attn(input_tensor)

    # They should be nearly identical
    assert torch.allclose(original_output[0], identity_output[0], atol=1e-5), "Identity attention output differs from original"


def test_compressed_attention_shape(attention_models):
    """Test if compressed attention maintains the correct output shape"""
    compressed_attn = attention_models["compressed"]

    # Create random input
    batch_size = 2
    sequence_length = 10
    hidden_size = attention_models["hidden_size"]

    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, sequence_length, hidden_size)

    # Get output
    with torch.no_grad():
        compressed_output = compressed_attn(input_tensor)

    # Check output shape
    assert compressed_output[0].shape == (batch_size, sequence_length, hidden_size), \
        f"Incorrect output shape: {compressed_output[0].shape}, expected {(batch_size, sequence_length, hidden_size)}"


def test_compress_decompress_cycle(attention_models):
    """Test compression followed by decompression recovers approximate input"""
    hidden_size = attention_models["hidden_size"]
    compressed_dim = attention_models["compressed_dim"]

    # Create DCT basis
    basis, inverse = dct.build_dct_basis(hidden_size, compressed_dim)

    # Create random input
    torch.manual_seed(42)
    input_tensor = torch.randn(1, hidden_size)

    # Compress
    compressed = dct.dct_project(input_tensor, basis)
    assert compressed.shape == (1, compressed_dim), f"Compressed shape incorrect: {compressed.shape}"

    # Decompress
    reconstructed = dct.dct_backproject(compressed, inverse)
    assert reconstructed.shape == (1, hidden_size), f"Reconstructed shape incorrect: {reconstructed.shape}"

    # Check reconstruction error (will be non-zero due to compression)
    error = torch.mean((input_tensor - reconstructed)**2)

    # Print error for debugging
    print(f"Compression ratio: {compressed_dim/hidden_size:.2f}")
    print(f"Reconstruction MSE: {error.item():.6f}")

    # Error should be reasonable for the compression level
    compression_ratio = compressed_dim / hidden_size
    max_expected_error = 0.5 * (1 - compression_ratio)**2

    assert error.item() < max_expected_error, f"Reconstruction error too high: {error.item():.6f} > {max_expected_error:.6f}"
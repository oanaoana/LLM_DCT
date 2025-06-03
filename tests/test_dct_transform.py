import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt

from transforms import dct

def test_dct_basis_orthogonality(input_dim=768):
    """Test if DCT basis is orthogonal for full rank"""
    basis, inverse = dct.build_dct_basis(input_dim, input_dim)
    product = torch.mm(basis, inverse)
    identity = torch.eye(input_dim, dtype=product.dtype, device=product.device)
    assert torch.allclose(product, identity, atol=1e-5), "DCT basis is not orthogonal"
    print(f"DCT orthogonality test passed for dim={input_dim}")

def plot_kv_dct_space(dim=768, num_vectors=64):
    """Generate random K and V, project to DCT space, and plot their spectra"""
    # Generate random K and V matrices (simulate transformer weights)
    K = torch.randn(num_vectors, dim)
    V = torch.randn(num_vectors, dim)
    # Build full DCT basis
    dct_basis, _ = dct.build_dct_basis(dim, dim)
    # Project to DCT space
    K_dct = torch.matmul(K, dct_basis)
    V_dct = torch.matmul(V, dct_basis)
    # Compute mean squared magnitude (energy) across vectors
    K_energy = (K_dct ** 2).mean(dim=0)
    V_energy = (V_dct ** 2).mean(dim=0)
    # Normalize
    K_energy /= K_energy.sum()
    V_energy /= V_energy.sum()
    # Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(K_energy.numpy(), label="K DCT Energy")
    plt.plot(V_energy.numpy(), label="V DCT Energy")
    plt.title("Normalized DCT Energy Spectrum")
    plt.xlabel("DCT Coefficient (Frequency)")
    plt.ylabel("Normalized Energy")
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.imshow(K_dct[:16].abs().numpy(), aspect='auto', cmap='viridis')
    plt.title("K (first 16 vectors) in DCT Space")
    plt.xlabel("DCT Coefficient")
    plt.ylabel("Vector Index")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("kv_dct_spectrum.png")
    plt.close()
    print("K and V DCT spectra plotted and saved as kv_dct_spectrum.png")

def plot_gpt2_kv_dct_space(model_name="gpt2", layer_indices=None, num_vectors=64):
    """Extract K and V from GPT-2, project to DCT space, and plot their spectra"""
    try:
        import transformers
    except ImportError:
        print("transformers library is required for this test. Install with 'pip install transformers'.")
        return

    # Load GPT-2 model
    model = transformers.GPT2Model.from_pretrained(model_name)
    config = model.config
    hidden_size = config.n_embd
    num_layers = config.n_layer

    # Select layers: first, middle, last by default
    if layer_indices is None:
        layer_indices = [0, num_layers // 2, num_layers - 1]

    dct_basis, _ = dct.build_dct_basis(hidden_size, hidden_size)

    for layer_idx in layer_indices:
        layer = model.h[layer_idx].attn
        # Extract K and V weights from the combined QKV matrix
        qkv_weight = layer.c_attn.weight  # [hidden_size, 3*hidden_size]
        k_weight = qkv_weight[:, hidden_size:2*hidden_size].detach().cpu()
        v_weight = qkv_weight[:, 2*hidden_size:].detach().cpu()

        # Use first num_vectors rows for analysis
        K = k_weight[:num_vectors, :]
        V = v_weight[:num_vectors, :]

        # Project to DCT space
        K_dct = torch.matmul(K, dct_basis)
        V_dct = torch.matmul(V, dct_basis)
        # Compute mean squared magnitude (energy) across vectors
        K_energy = (K_dct ** 2).mean(dim=0)
        V_energy = (V_dct ** 2).mean(dim=0)
        # Normalize
        K_energy /= K_energy.sum()
        V_energy /= V_energy.sum()
        # Plot
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(K_energy.numpy(), label="K DCT Energy")
        plt.plot(V_energy.numpy(), label="V DCT Energy")
        plt.title(f"Layer {layer_idx}: Normalized DCT Energy Spectrum")
        plt.xlabel("DCT Coefficient (Frequency)")
        plt.ylabel("Normalized Energy")
        plt.legend()
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.imshow(K_dct[:16].abs().numpy(), aspect='auto', cmap='viridis')
        plt.title(f"Layer {layer_idx}: K (first 16 vectors) in DCT Space")
        plt.xlabel("DCT Coefficient")
        plt.ylabel("Vector Index")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"gpt2_layer{layer_idx}_kv_dct_spectrum.png")
        plt.close()
        print(f"Layer {layer_idx}: K and V DCT spectra plotted and saved as gpt2_layer{layer_idx}_kv_dct_spectrum.png")

if __name__ == "__main__":
    test_dct_basis_orthogonality()
    plot_kv_dct_space()
    plot_gpt2_kv_dct_space()
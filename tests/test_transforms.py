"""
Tests for transform implementations (DCT, PSWF)
"""
import torch
import pytest
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transforms import dct, pswf

# Set to True to also test PSWF, False to test only DCT
TEST_PSWF = False

@pytest.mark.parametrize("input_dim", [768, 1024])
@pytest.mark.parametrize("compression_ratio", [1.0, 0.75, 0.5])
def test_dct_basis_orthogonality(input_dim, compression_ratio):
    """Test if DCT basis is orthogonal"""
    compressed_dim = int(input_dim * compression_ratio)

    # Build basis
    basis, inverse = dct.build_dct_basis(input_dim, compressed_dim)

    # Check basis shape
    assert basis.shape == (input_dim, compressed_dim), f"Basis shape incorrect: {basis.shape}"
    assert inverse.shape == (compressed_dim, input_dim), f"Inverse shape incorrect: {inverse.shape}"

    # For full rank, basis should be orthogonal
    if compression_ratio == 1.0:
        # B * B^T should be close to identity
        product = torch.mm(basis, inverse)
        identity = torch.eye(input_dim)
        assert torch.allclose(product, identity, atol=1e-5), "Basis is not orthogonal"


@pytest.mark.parametrize("input_dim", [768, 1024])
@pytest.mark.parametrize("compression_ratio", [1.0, 0.75, 0.5, 0.25])
def test_dct_reconstruction(input_dim, compression_ratio):
    """Test DCT projection and backprojection"""
    compressed_dim = int(input_dim * compression_ratio)

    # Build basis
    basis, inverse = dct.build_dct_basis(input_dim, compressed_dim)

    # Create random input
    x = torch.randn(1, input_dim)

    # Project and backproject
    x_compressed = dct.dct_project(x, basis)
    x_reconstructed = dct.dct_backproject(x_compressed, inverse)

    # Check shape
    assert x_compressed.shape == (1, compressed_dim), f"Compressed shape wrong: {x_compressed.shape}"
    assert x_reconstructed.shape == (1, input_dim), f"Reconstructed shape wrong: {x_reconstructed.shape}"

    # For full rank, reconstruction should be perfect
    if compression_ratio == 1.0:
        assert torch.allclose(x, x_reconstructed, atol=1e-5), "Perfect reconstruction failed"
    else:
        # Check reconstruction quality (should be better with higher compression ratio)
        error = torch.mean((x - x_reconstructed) ** 2).item()

        # Check relative error is proportional to compression
        # Less compression should mean less error
        assert error < (1.0 - compression_ratio), f"Reconstruction error too high: {error}"


@pytest.mark.skipif(not TEST_PSWF, reason="PSWF tests disabled")
@pytest.mark.parametrize("input_dim", [768, 1024])
@pytest.mark.parametrize("compression_ratio", [1.0, 0.75, 0.5, 0.25])
def test_pswf_basis(input_dim, compression_ratio):
    """Test PSWF basis properties"""
    compressed_dim = int(input_dim * compression_ratio)

    # Build basis
    basis, inverse = pswf.build_pswf_basis(input_dim, compressed_dim)

    # Check basis shape
    assert basis.shape == (input_dim, compressed_dim), f"Basis shape incorrect: {basis.shape}"
    assert inverse.shape == (compressed_dim, input_dim), f"Inverse shape incorrect: {inverse.shape}"


@pytest.mark.skipif(not TEST_PSWF, reason="PSWF tests disabled")
@pytest.mark.parametrize("input_dim", [768, 1024])
@pytest.mark.parametrize("compression_ratio", [1.0, 0.75, 0.5, 0.25])
def test_transform_comparison(input_dim, compression_ratio):
    """Compare DCT and PSWF reconstruction quality"""
    compressed_dim = int(input_dim * compression_ratio)

    # Build bases
    dct_basis, dct_inverse = dct.build_dct_basis(input_dim, compressed_dim)
    pswf_basis, pswf_inverse = pswf.build_pswf_basis(input_dim, compressed_dim)

    # Create random input with some structure (not just noise)
    x = torch.zeros(1, input_dim)
    for i in range(5):
        freq = np.random.randint(1, 20)
        phase = np.random.random() * np.pi
        x += torch.sin(torch.linspace(0, freq * np.pi, input_dim) + phase).unsqueeze(0)

    # Project and backproject with both transforms
    x_dct = dct.dct_project(x, dct_basis)
    x_dct_reconstructed = dct.dct_backproject(x_dct, dct_inverse)

    x_pswf = pswf.pswf_project(x, pswf_basis)
    x_pswf_reconstructed = pswf.pswf_backproject(x_pswf, pswf_inverse)

    # Calculate reconstruction errors
    dct_error = torch.mean((x - x_dct_reconstructed) ** 2).item()
    pswf_error = torch.mean((x - x_pswf_reconstructed) ** 2).item()

    # Both should have reasonable errors proportional to compression
    assert dct_error < (1.0 - compression_ratio) * 2, f"DCT error too high: {dct_error}"
    assert pswf_error < (1.0 - compression_ratio) * 2, f"PSWF error too high: {pswf_error}"

    # Log comparative performance
    print(f"Input dim: {input_dim}, Compression: {compression_ratio}")
    print(f"DCT error: {dct_error:.6f}, PSWF error: {pswf_error:.6f}")


def visualize_attention_in_dct_space(seq_len=10, dim=768, compression_ratio=0.5, attention_pattern="random"):
    """
    Visualize how attention patterns look in DCT frequency domain

    Args:
        seq_len: Sequence length for attention
        dim: Embedding dimension
        compression_ratio: Compression amount
        attention_pattern: Type of attention pattern to visualize
    """
    compressed_dim = int(dim * compression_ratio)

    # Build DCT basis
    basis, inverse = dct.build_dct_basis(dim, compressed_dim)

    # Create artificial attention patterns
    if attention_pattern == "random":
        # Random attention
        attention_scores = torch.rand(1, seq_len, seq_len)
    elif attention_pattern == "local":
        # Local attention (focused on diagonal)
        attention_scores = torch.zeros(1, seq_len, seq_len)
        for i in range(seq_len):
            for j in range(seq_len):
                attention_scores[0, i, j] = torch.exp(torch.tensor(-abs(i-j) / 2.0))
    elif attention_pattern == "global":
        # Global attention (one token attends to all)
        attention_scores = torch.zeros(1, seq_len, seq_len)
        attention_scores[0, 0, :] = 1.0  # First token attends to all
    else:
        raise ValueError(f"Unknown attention pattern: {attention_pattern}")

    # Normalize with softmax
    attention_probs = torch.softmax(attention_scores, dim=-1)

    # Create random Q, K, V
    query = torch.randn(1, seq_len, dim)
    key = torch.randn(1, seq_len, dim)
    value = torch.randn(1, seq_len, dim)

    # Transform K, V to DCT space
    key_flat = key.reshape(-1, dim)
    value_flat = value.reshape(-1, dim)

    key_dct = dct.dct_project(key_flat, basis).reshape(1, seq_len, compressed_dim)
    value_dct = dct.dct_project(value_flat, basis).reshape(1, seq_len, compressed_dim)

    # Calculate attention output in original space
    attention_output = torch.bmm(attention_probs, value)

    # Calculate attention output in DCT space
    attention_output_dct = torch.bmm(attention_probs, value_dct)

    # Calculate sparsity in DCT space
    # For each sequence position, measure coefficient distribution
    dct_magnitude = torch.abs(value_dct)
    dct_energy = torch.sum(dct_magnitude**2, dim=0)  # Sum over batch
    dct_energy_normalized = dct_energy / torch.sum(dct_energy, dim=1, keepdim=True)

    # Measure attention output sparsity in DCT space
    output_dct_magnitude = torch.abs(attention_output_dct)
    output_dct_energy = torch.sum(output_dct_magnitude**2, dim=0)
    output_dct_energy_normalized = output_dct_energy / torch.sum(output_dct_energy, dim=1, keepdim=True)

    # Create visualization
    plt.figure(figsize=(18, 12))

    # Plot attention pattern
    plt.subplot(2, 3, 1)
    sns.heatmap(attention_probs[0].numpy(), cmap="viridis")
    plt.title(f"Attention Pattern ({attention_pattern})")

    # Plot DCT coefficient magnitudes for value vectors
    plt.subplot(2, 3, 2)
    sns.heatmap(dct_magnitude[0].numpy(), cmap="plasma")
    plt.title("Value Vectors in DCT Space")

    # Plot DCT energy distribution (averaged over sequence)
    plt.subplot(2, 3, 3)
    plt.plot(torch.mean(dct_energy_normalized, dim=0).numpy())
    plt.title("Average DCT Energy Distribution")
    plt.xlabel("DCT Coefficient")
    plt.ylabel("Normalized Energy")
    plt.grid(True)

    # Plot attention output in DCT space
    plt.subplot(2, 3, 4)
    sns.heatmap(output_dct_magnitude[0].numpy(), cmap="plasma")
    plt.title("Attention Output in DCT Space")

    # Plot attention output DCT energy distribution
    plt.subplot(2, 3, 5)
    plt.plot(torch.mean(output_dct_energy_normalized, dim=0).numpy())
    plt.title("Attention Output DCT Energy")
    plt.xlabel("DCT Coefficient")
    plt.ylabel("Normalized Energy")
    plt.grid(True)

    # Plot cumulative energy to show sparsity
    plt.subplot(2, 3, 6)
    cumulative_energy = torch.cumsum(torch.mean(dct_energy_normalized, dim=0), dim=0).numpy()
    cumulative_output_energy = torch.cumsum(torch.mean(output_dct_energy_normalized, dim=0), dim=0).numpy()

    plt.plot(cumulative_energy, label="Value Vectors")
    plt.plot(cumulative_output_energy, label="Attention Output")

    # Find 90% energy point
    energy_90_value = np.argmax(cumulative_energy >= 0.9) / compressed_dim
    energy_90_output = np.argmax(cumulative_output_energy >= 0.9) / compressed_dim

    plt.axhline(0.9, color='gray', linestyle='--')
    plt.axvline(energy_90_value * compressed_dim, color='blue', linestyle='--',
               label=f"90% Energy at {energy_90_value*100:.1f}%")
    plt.axvline(energy_90_output * compressed_dim, color='orange', linestyle='--',
               label=f"90% Output Energy at {energy_90_output*100:.1f}%")

    plt.title("Cumulative Energy")
    plt.xlabel("DCT Coefficient")
    plt.ylabel("Cumulative Energy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"attention_dct_sparsity_{attention_pattern}.png")
    plt.close()

    print(f"Visualization saved as: attention_dct_sparsity_{attention_pattern}.png")

    # Return sparsity metrics
    return {
        "value_90pct_energy_at": energy_90_value,
        "output_90pct_energy_at": energy_90_output,
        "compression_ratio": compression_ratio
    }


def test_attention_sparsity():
    """Test and visualize attention sparsity in DCT space"""
    # Test with different attention patterns
    patterns = ["random", "local", "global"]
    dims = [768]
    compression_ratios = [0.5]

    results = {}

    for pattern in patterns:
        pattern_results = {}
        for dim in dims:
            for ratio in compression_ratios:
                metrics = visualize_attention_in_dct_space(
                    seq_len=12,
                    dim=dim,
                    compression_ratio=ratio,
                    attention_pattern=pattern
                )
                pattern_results[f"dim{dim}_ratio{ratio}"] = metrics

        results[pattern] = pattern_results

    # Print summary
    print("\n=== Attention Sparsity in DCT Space ===")
    for pattern, pattern_results in results.items():
        print(f"\nPattern: {pattern}")
        for config, metrics in pattern_results.items():
            print(f"  {config}: 90% Energy at {metrics['value_90pct_energy_at']*100:.1f}% coefficients")
            print(f"  {config}: 90% Output Energy at {metrics['output_90pct_energy_at']*100:.1f}% coefficients")


def test_qkv_selective_compression():
    """Test selective compression of Q, K, V with different ratios"""
    # Settings
    batch_size = 2
    seq_len = 10
    dim = 768

    # Compression ratios
    q_ratio = 1.0  # No compression for Q
    k_ratio = 0.5  # 50% compression for K
    v_ratio = 0.5  # 50% compression for V

    # Create random Q, K, V
    query = torch.randn(batch_size, seq_len, dim)
    key = torch.randn(batch_size, seq_len, dim)
    value = torch.randn(batch_size, seq_len, dim)

    # Build DCT bases for each
    q_basis, q_inverse = dct.build_dct_basis(dim, int(dim * q_ratio))
    k_basis, k_inverse = dct.build_dct_basis(dim, int(dim * k_ratio))
    v_basis, v_inverse = dct.build_dct_basis(dim, int(dim * v_ratio))

    # Compress K and V
    key_flat = key.reshape(-1, dim)
    value_flat = value.reshape(-1, dim)

    key_compressed = dct.dct_project(key_flat, k_basis)
    value_compressed = dct.dct_project(value_flat, v_basis)

    # Reshape back
    key_compressed = key_compressed.reshape(batch_size, seq_len, int(dim * k_ratio))
    value_compressed = value_compressed.reshape(batch_size, seq_len, int(dim * v_ratio))

    # For attention computation, decompress K and V
    key_reconstructed = dct.dct_backproject(key_compressed.reshape(-1, int(dim * k_ratio)), k_inverse)
    value_reconstructed = dct.dct_backproject(value_compressed.reshape(-1, int(dim * v_ratio)), v_inverse)

    # Reshape reconstructed K and V
    key_reconstructed = key_reconstructed.reshape(batch_size, seq_len, dim)
    value_reconstructed = value_reconstructed.reshape(batch_size, seq_len, dim)

    # Calculate attention with original and reconstructed K, V
    scores_original = torch.bmm(query, key.transpose(1, 2)) / (dim ** 0.5)
    scores_reconstructed = torch.bmm(query, key_reconstructed.transpose(1, 2)) / (dim ** 0.5)

    # Apply softmax
    attn_original = torch.softmax(scores_original, dim=-1)
    attn_reconstructed = torch.softmax(scores_reconstructed, dim=-1)

    # Calculate outputs
    output_original = torch.bmm(attn_original, value)
    output_reconstructed = torch.bmm(attn_reconstructed, value_reconstructed)

    # Calculate errors
    attn_error = torch.mean((attn_original - attn_reconstructed) ** 2).item()
    output_error = torch.mean((output_original - output_reconstructed) ** 2).item()

    # Calculate memory savings
    original_memory = batch_size * seq_len * dim * 3  # Q, K, V
    compressed_memory = batch_size * seq_len * (dim + int(dim * k_ratio) + int(dim * v_ratio))
    memory_savings = 1 - (compressed_memory / original_memory)

    print("\n=== Selective QKV Compression ===")
    print(f"Q compression: {q_ratio:.2f}, K compression: {k_ratio:.2f}, V compression: {v_ratio:.2f}")
    print(f"Attention matrix error: {attn_error:.6f}")
    print(f"Output error: {output_error:.6f}")
    print(f"Memory savings: {memory_savings*100:.1f}%")

    # Visualize attention patterns
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.heatmap(attn_original[0].numpy(), cmap="viridis")
    plt.title("Original Attention")

    plt.subplot(1, 3, 2)
    sns.heatmap(attn_reconstructed[0].numpy(), cmap="viridis")
    plt.title("Reconstructed Attention")

    plt.subplot(1, 3, 3)
    error_map = torch.abs(attn_original - attn_reconstructed)[0].numpy()
    sns.heatmap(error_map, cmap="Reds")
    plt.title(f"Error (MSE: {attn_error:.6f})")

    plt.tight_layout()
    plt.savefig("selective_qkv_compression.png")
    plt.close()

    print("Visualization saved as: selective_qkv_compression.png")

    return {
        "attn_error": attn_error,
        "output_error": output_error,
        "memory_savings": memory_savings
    }


def analyze_gpt2_kv_patterns(model_size="small", layer_indices=None, visualize=True):
    """
    Analyze K and V matrices from GPT-2 model in DCT space

    Args:
        model_size: GPT-2 model size ("small", "medium", "large", "xl")
        layer_indices: Which layers to analyze (None for auto selection)
        visualize: Whether to create visualizations

    Returns:
        Dictionary with analysis results
    """
    print(f"\n=== Analyzing GPT-2-{model_size} K and V matrices in DCT space ===")

    try:
        import transformers

        # Map model size to actual model name
        model_map = {
            "small": "gpt2",
            "medium": "gpt2-medium",
            "large": "gpt2-large",
            "xl": "gpt2-xl"
        }

        model_name = model_map.get(model_size, "gpt2")

        # Load model
        print(f"Loading {model_name}...")
        model = transformers.GPT2Model.from_pretrained(model_name)
        config = model.config

        # Get model dimensions
        num_layers = config.n_layer
        hidden_size = config.n_embd

        print(f"Model has {num_layers} layers with hidden size {hidden_size}")

        # If no layer indices provided, pick representative layers
        if layer_indices is None:
            # First, middle, and last layers
            layer_indices = [0, num_layers//2, num_layers-1]

        # Extract K and V matrices from specified layers
        k_matrices = []
        v_matrices = []
        layer_names = []

        for layer_idx in layer_indices:
            # Get appropriate layer
            layer = model.h[layer_idx].attn

            # GPT-2 has combined QKV matrix, need to extract K and V parts
            if hasattr(layer, 'c_attn'):
                # GPT-2 style - combined QKV weights
                qkv_weight = layer.c_attn.weight

                # The weight matrix is [hidden_size, 3*hidden_size]
                # Split into Q, K, V parts
                k_weight = qkv_weight[:, hidden_size:2*hidden_size]
                v_weight = qkv_weight[:, 2*hidden_size:]
            else:
                # Separate K, V weights
                k_weight = layer.k_proj.weight
                v_weight = layer.v_proj.weight

            k_matrices.append(k_weight.detach())
            v_matrices.append(v_weight.detach())
            layer_names.append(f"Layer {layer_idx}")

        # Build DCT basis for the hidden size
        dct_basis, dct_inverse = dct.build_dct_basis(hidden_size, hidden_size)

        # Transform K, V matrices to DCT space
        k_dct_spectra = []
        v_dct_spectra = []

        for k_mat, v_mat in zip(k_matrices, v_matrices):
            # For each row of K, transform to DCT space
            k_rows = []
            for i in range(min(100, k_mat.shape[0])):  # Limit to 100 rows for speed
                k_row = k_mat[i:i+1]
                k_dct = dct.dct_project(k_row, dct_basis)
                k_rows.append(k_dct)

            # Same for V
            v_rows = []
            for i in range(min(100, v_mat.shape[0])):
                v_row = v_mat[i:i+1]
                v_dct = dct.dct_project(v_row, dct_basis)
                v_rows.append(v_dct)

            # Stack all rows
            k_dct_spectrum = torch.cat(k_rows, dim=0)
            v_dct_spectrum = torch.cat(v_rows, dim=0)

            k_dct_spectra.append(k_dct_spectrum)
            v_dct_spectra.append(v_dct_spectrum)

        # Analyze results
        results = {}

        # For each compression ratio, calculate energy retention
        compression_ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

        # Create figure if visualizing
        if visualize:
            plt.figure(figsize=(20, 6 * len(layer_indices)))

        # Analyze each layer
        for layer_idx, layer_name in enumerate(layer_names):
            layer_results = {}

            # Get DCT spectra
            k_dct = k_dct_spectra[layer_idx]
            v_dct = v_dct_spectra[layer_idx]

            # Calculate energy
            k_energy = torch.mean(torch.abs(k_dct)**2, dim=0)
            v_energy = torch.mean(torch.abs(v_dct)**2, dim=0)

            # Normalize
            k_energy_norm = k_energy / torch.sum(k_energy)
            v_energy_norm = v_energy / torch.sum(v_energy)

            # Calculate cumulative energy
            k_cum_energy = torch.cumsum(k_energy_norm, dim=0)
            v_cum_energy = torch.cumsum(v_energy_norm, dim=0)

            # For each compression ratio, find energy retention
            k_retention = {}
            v_retention = {}

            for ratio in compression_ratios:
                keep_coeffs = int(hidden_size * ratio)
                k_retained = k_cum_energy[keep_coeffs-1].item() * 100
                v_retained = v_cum_energy[keep_coeffs-1].item() * 100

                k_retention[ratio] = k_retained
                v_retention[ratio] = v_retained

                print(f"{layer_name} - Compression ratio {ratio:.1f}: K energy {k_retained:.2f}%, V energy {v_retained:.2f}%")

            # Find optimal compression
            k_optimal = min([r for r in compression_ratios if k_retention[r] >= 95], default=0.9)
            v_optimal = min([r for r in compression_ratios if v_retention[r] >= 95], default=0.9)

            print(f"{layer_name} - 95% energy optimal compression: K={k_optimal:.1f}, V={v_optimal:.1f}")

            # Store results
            layer_results["k_retention"] = k_retention
            layer_results["v_retention"] = v_retention
            layer_results["k_optimal"] = k_optimal
            layer_results["v_optimal"] = v_optimal

            # Visualize if requested
            if visualize:
                # Plot energy distribution
                ax1 = plt.subplot(len(layer_indices), 3, layer_idx * 3 + 1)
                ax1.plot(k_energy_norm.numpy(), label='K')
                ax1.plot(v_energy_norm.numpy(), label='V')
                ax1.set_title(f"{layer_name}: DCT Coefficient Energy")
                ax1.set_xlabel("DCT Coefficient (Frequency)")
                ax1.set_ylabel("Normalized Energy")
                ax1.legend()
                ax1.grid(True)

                # Plot cumulative energy
                ax2 = plt.subplot(len(layer_indices), 3, layer_idx * 3 + 2)
                ax2.plot(k_cum_energy.numpy(), label='K')
                ax2.plot(v_cum_energy.numpy(), label='V')

                # Find 90%, 95%, and 99% energy points
                for energy_threshold, color, style in [(0.9, 'green', '--'), (0.95, 'orange', '-.'), (0.99, 'red', ':')]:
                    k_threshold_idx = torch.where(k_cum_energy >= energy_threshold)[0][0].item()
                    v_threshold_idx = torch.where(v_cum_energy >= energy_threshold)[0][0].item()

                    ax2.axhline(energy_threshold, color=color, linestyle=style, alpha=0.5)
                    ax2.axvline(k_threshold_idx, color='blue', linestyle=style, alpha=0.5)
                    ax2.axvline(v_threshold_idx, color='green', linestyle=style, alpha=0.5)

                ax2.set_title(f"{layer_name}: Cumulative Energy")
                ax2.set_xlabel("DCT Coefficient (Frequency)")
                ax2.set_ylabel("Cumulative Energy")
                ax2.legend()
                ax2.grid(True)

                # Plot heatmap of coefficients
                ax3 = plt.subplot(len(layer_indices), 3, layer_idx * 3 + 3)
                sns.heatmap(torch.abs(k_dct[:20]).numpy(), cmap='viridis', ax=ax3)
                ax3.set_title(f"{layer_name}: K Matrix DCT Coefficients")
                ax3.set_xlabel("DCT Coefficient (Frequency)")
                ax3.set_ylabel("Row Index")

            results[layer_name] = layer_results

        # Compute model-wide averages
        avg_k_retention = {ratio: np.mean([results[l]["k_retention"][ratio] for l in layer_names]) for ratio in compression_ratios}
        avg_v_retention = {ratio: np.mean([results[l]["v_retention"][ratio] for l in layer_names]) for ratio in compression_ratios}

        # Find model-wide optimal compression
        model_k_optimal = min([r for r in compression_ratios if avg_k_retention[r] >= 95], default=0.9)
        model_v_optimal = min([r for r in compression_ratios if avg_v_retention[r] >= 95], default=0.9)

        print(f"\nModel-wide 95% energy optimal compression: K={model_k_optimal:.1f}, V={model_v_optimal:.1f}")

        results["model_avg"] = {
            "k_retention": avg_k_retention,
            "v_retention": avg_v_retention,
            "k_optimal": model_k_optimal,
            "v_optimal": model_v_optimal
        }

        if visualize:
            # Create model-wide recommendation plot
            plt.tight_layout()
            plt.savefig(f"gpt2_{model_size}_kv_dct_analysis.png")
            plt.close()

            # Additional plot for compression recommendations
            plt.figure(figsize=(15, 10))

            plt.subplot(2, 1, 1)
            for layer_idx, layer_name in enumerate(layer_names):
                k_retention = results[layer_name]["k_retention"]
                plt.plot(compression_ratios, [k_retention[r] for r in compression_ratios],
                         'o-', label=f"{layer_name} K")

            plt.plot(compression_ratios, [avg_k_retention[r] for r in compression_ratios],
                     'o-', linewidth=3, label="Model Avg K")

            plt.axhline(95, color='red', linestyle='--', label='95% Energy Target')
            plt.xlabel("Compression Ratio")
            plt.ylabel("Energy Retention (%)")
            plt.title(f"GPT-2-{model_size}: K Matrix Energy Retention by Compression Ratio")
            plt.grid(True)
            plt.legend()

            plt.subplot(2, 1, 2)
            for layer_idx, layer_name in enumerate(layer_names):
                v_retention = results[layer_name]["v_retention"]
                plt.plot(compression_ratios, [v_retention[r] for r in compression_ratios],
                         'o-', label=f"{layer_name} V")

            plt.plot(compression_ratios, [avg_v_retention[r] for r in compression_ratios],
                     'o-', linewidth=3, label="Model Avg V")

            plt.axhline(95, color='red', linestyle='--', label='95% Energy Target')
            plt.xlabel("Compression Ratio")
            plt.ylabel("Energy Retention (%)")
            plt.title(f"GPT-2-{model_size}: V Matrix Energy Retention by Compression Ratio")
            plt.grid(True)
            plt.legend()

            plt.tight_layout()
            plt.savefig(f"gpt2_{model_size}_compression_recommendations.png")
            plt.close()

            print(f"Visualizations saved as: gpt2_{model_size}_kv_dct_analysis.png and gpt2_{model_size}_compression_recommendations.png")

        return results

    except ImportError as e:
        print(f"Could not import transformers library: {e}")
        print("Please install with: pip install transformers")
        return None
    except Exception as e:
        print(f"Error analyzing GPT-2 model: {e}")
        return None

if __name__ == "__main__":
    print("Running DCT and attention visualization tests")

    # Add the GPT-2 specific analysis
    try:
        gpt2_results = analyze_gpt2_kv_patterns(model_size="small")
        if gpt2_results:
            print(f"\nGPT-2 Recommended K compression ratio: {gpt2_results['model_avg']['k_optimal']}")
            print(f"GPT-2 Recommended V compression ratio: {gpt2_results['model_avg']['v_optimal']}")
    except Exception as e:
        print(f"GPT-2 analysis failed: {e}")

    # Run the other tests too
    test_attention_sparsity()
    test_qkv_selective_compression()
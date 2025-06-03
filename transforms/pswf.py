import torch
import math

def pswf_project(x, pswf_basis):
    """
    Project input onto PSWF basis

    Args:
        x: Input tensor of shape (batch, dim)
        pswf_basis: PSWF basis of shape (dim, compressed_dim)

    Returns:
        Projected tensor of shape (batch, compressed_dim)
    """
    # Check input dimensions
    if x.dim() == 1:
        x = x.unsqueeze(0)  # Add batch dimension if missing

    assert x.shape[1] == pswf_basis.shape[0], \
        f"Input dim {x.shape[1]} doesn't match basis dim {pswf_basis.shape[0]}"

    # Ensure x is on the correct device
    if pswf_basis.device != x.device:
        pswf_basis = pswf_basis.to(x.device)

    # Ensure consistent dtype
    if pswf_basis.dtype != x.dtype:
        pswf_basis = pswf_basis.to(dtype=x.dtype)

    # Project onto basis
    return torch.matmul(x, pswf_basis)

def pswf_backproject(x, pswf_inverse):
    """
    Project back from compressed space to original space

    Args:
        x: Tensor in compressed space of shape (batch, compressed_dim)
        pswf_inverse: Backprojection matrix of shape (compressed_dim, dim)

    Returns:
        Reconstructed tensor of shape (batch, dim)
    """
    # Check if the inverse matrix has the right shape
    compressed_dim = x.shape[1]

    # Ensure correct orientation - we expect (compressed_dim, N)
    if pswf_inverse.shape[0] != compressed_dim:
        # If wrong orientation, reshape it (but print warning)
        print(f"Warning: Inverse matrix has wrong shape: {pswf_inverse.shape}, expected first dim to be {compressed_dim}")
        pswf_inverse = pswf_inverse.T

    # Ensure x is on the correct device
    if pswf_inverse.device != x.device:
        pswf_inverse = pswf_inverse.to(x.device)

    # Ensure consistent dtype
    if pswf_inverse.dtype != x.dtype:
        pswf_inverse = pswf_inverse.to(dtype=x.dtype)

    # Back-project via matrix multiplication
    return torch.matmul(x, pswf_inverse)

def build_pswf_basis(N, compressed_dim, bandwidth=math.pi, use_svd=True, eps=1e-10, dtype=torch.float64):
    """
    Build PSWF basis for projection and back-projection with improved stability

    Args:
        N: Original dimension
        compressed_dim: Target compressed dimension
        bandwidth: PSWF bandwidth parameter (default: π)
        use_svd: Whether to use SVD instead of eigendecomposition for added stability
        eps: Small value for numerical stability
        dtype: Data type to use for computation (default: float64/double)

    Returns:
        Tuple of (basis, pseudo_inverse)
    """
    # Input validation
    assert compressed_dim <= N, f"Compressed dimension {compressed_dim} cannot exceed original dimension {N}"
    if compressed_dim == N:
        # Return identity matrices for no compression
        return torch.eye(N, dtype=dtype), torch.eye(N, dtype=dtype)

    # Create discretization grid in double precision
    x = torch.linspace(-1, 1, N, dtype=dtype)
    dx = x[1] - x[0]

    # Build kernel matrix - sinc kernel corresponds to PSWFs
    X, Y = torch.meshgrid(x, x, indexing='ij')
    kernel = torch.sinc(bandwidth * (X - Y) / math.pi) * dx

    # Add a small amount to diagonal for stability
    kernel = kernel + eps * torch.eye(N, dtype=dtype)

    # Force kernel to be exactly symmetric
    kernel = 0.5 * (kernel + kernel.T)

    # Try progressively more stable methods until we get a good result
    basis = None
    pseudo_inverse = None
    reconstruction_error = float('inf')

    # First try SVD if requested
    if use_svd:
        try:
            # Use double precision for better numerical stability
            kernel_double = kernel.double()

            # Use the more stable torch.linalg.svd function
            U, S, Vh = torch.linalg.svd(kernel_double, full_matrices=False)

            # Threshold very small singular values to avoid numerical instability
            S_threshold = torch.max(S) * eps * 10
            #condition_number = S[0] / (S[-1] + eps)
            #lambda_reg = eps * condition_number
            #S_inv_diag = S[:compressed_dim] / (S[:compressed_dim]**2 + lambda_reg)

            S = torch.where(S < S_threshold, torch.zeros_like(S), S)

            # Keep only the top compressed_dim singular vectors for basis
            basis = U[:, :compressed_dim].float()

            # Create diagonal matrix of singular values for pseudoinverse
            S_inv_diag = torch.zeros_like(S[:compressed_dim])
            S_inv_diag = torch.where(S[:compressed_dim] > S_threshold,
                                    1.0 / S[:compressed_dim],
                                    torch.zeros_like(S[:compressed_dim]))

            # Create pseudoinverse with correct dimensions (compressed_dim, N)
            pseudo_inverse = torch.mm(
                torch.diag(S_inv_diag.float()),
                U[:, :compressed_dim].T.float()
            )

            # Test reconstruction error
            test_vec = torch.ones(1, N, device=kernel.device)
            compressed = torch.matmul(test_vec, basis)
            reconstructed = torch.matmul(compressed, pseudo_inverse)
            reconstruction_error = torch.mean((test_vec - reconstructed) ** 2).item()

            if reconstruction_error > 1.0:
                print(f"SVD method gave high error: {reconstruction_error:.6f}, trying eigendecomposition")
                basis = None  # Reset to try next method

        except Exception as e:
            print(f"SVD failed: {str(e)}. Trying eigendecomposition.")
            basis = None  # Reset to try next method

    # If SVD failed or gave poor results, try eigendecomposition
    if basis is None:
        try:
            # Use double precision for better numerical stability
            kernel_double = kernel.double()

            # Use the more stable symmetric eigensolver
            eigenvalues, eigenvectors = torch.linalg.eigh(kernel_double)

            # Sort by decreasing eigenvalue
            idx = torch.argsort(eigenvalues, descending=True)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Apply a threshold to eigenvalues to avoid numerical instability
            eigen_threshold = torch.max(torch.abs(eigenvalues)) * eps * 10
            stable_eigenvalues = eigenvalues.clone()
            stable_eigenvalues[torch.abs(eigenvalues) < eigen_threshold] = 0.0

            # Select top eigenvectors for basis (ensure float32)
            basis = eigenvectors[:, :compressed_dim].float()

            # Compute pseudoinverse with stabilized eigenvalues (ensure correct dtype)
            pseudo_inverse = torch.zeros((compressed_dim, N), device=basis.device, dtype=torch.float32)

            for i in range(compressed_dim):
                if torch.abs(stable_eigenvalues[i]) > eigen_threshold:
                    # Convert to float32 before assigning
                    pseudo_inverse[i] = (eigenvectors[:, i] / stable_eigenvalues[i]).float()

            # Test reconstruction error
            test_vec = torch.ones(1, N, device=kernel.device)
            compressed = torch.matmul(test_vec, basis)
            reconstructed = torch.matmul(compressed, pseudo_inverse)
            reconstruction_error = torch.mean((test_vec - reconstructed) ** 2).item()

            if reconstruction_error > 1.0:
                print(f"Eigendecomposition gave high error: {reconstruction_error:.6f}, trying regularized approach")

                # Try regularized approach - add progressively more regularization
                for reg_factor in [1e-4, 1e-3, 1e-2, 1e-1]:
                    # Explicitly compute pseudoinverse with regularization
                    basis_t = basis.T.float()  # Ensure float32
                    reg_matrix = torch.eye(compressed_dim, device=basis.device, dtype=torch.float32) * reg_factor

                    # (B^T B + λI)^(-1) B^T - ensure same dtype
                    basis_product = torch.mm(basis_t, basis.float())
                    regularized = basis_product + reg_matrix

                    try:
                        # Solve with consistent dtype
                        pseudo_inverse = torch.linalg.solve(regularized, basis_t)

                        # Test reconstruction again
                        test_vec = torch.ones(1, N, device=kernel.device)
                        compressed = torch.matmul(test_vec, basis)
                        reconstructed = torch.matmul(compressed, pseudo_inverse)
                        reconstruction_error = torch.mean((test_vec - reconstructed) ** 2).item()

                        if reconstruction_error <= 1.0:
                            print(f"Regularized pseudoinverse succeeded with factor {reg_factor}, error: {reconstruction_error:.6f}")
                            break
                    except Exception as e:
                        print(f"Regularization with factor {reg_factor} failed: {str(e)}")

        except Exception as e:
            print(f"Eigendecomposition failed: {str(e)}. Using direct QR approach.")

            # Last resort: QR factorization to create orthogonal basis
            random_basis = torch.randn(N, compressed_dim, device=kernel.device, dtype=torch.float32)
            basis, _ = torch.linalg.qr(random_basis)

            # Compute a stable pseudoinverse directly with strong regularization
            basis_t = basis.T  # Already float32
            reg_matrix = torch.eye(compressed_dim, device=basis.device, dtype=torch.float32) * 1e-2

            try:
                # Use consistent dtype
                pseudo_inverse = torch.linalg.solve(
                    torch.mm(basis_t, basis) + reg_matrix,
                    basis_t
                )

                # Test reconstruction one more time
                test_vec = torch.ones(1, N, device=kernel.device)
                compressed = torch.matmul(test_vec, basis)
                reconstructed = torch.matmul(compressed, pseudo_inverse)
                reconstruction_error = torch.mean((test_vec - reconstructed) ** 2).item()
            except Exception as e:
                print(f"QR approach failed: {str(e)}. Using Moore-Penrose pseudoinverse.")

                # Absolute last resort: direct pseudoinverse
                pseudo_inverse = torch.linalg.pinv(basis.T, rcond=1e-2)

    # Final check - if all else failed, print diagnostic information and retry with high regularization
    if reconstruction_error > 1.0:
        # Show detailed diagnostic information
        print(f"WARNING: All methods failed to achieve good reconstruction. Error: {reconstruction_error:.6f}")

        # Apply an aggressive regularization to the pseudo-inverse
        try:
            basis_t = basis.T
            strong_reg = torch.eye(compressed_dim, device=basis.device, dtype=torch.float32) * 0.5
            pseudo_inverse = torch.linalg.solve(
                torch.mm(basis_t, basis) + strong_reg,
                basis_t
            )

            # Final stability check
            test_vec = torch.ones(1, N, device=kernel.device)
            compressed = torch.matmul(test_vec, basis)
            reconstructed = torch.matmul(compressed, pseudo_inverse)
            reconstruction_error = torch.mean((test_vec - reconstructed) ** 2).item()
            print(f"After aggressive regularization, reconstruction error: {reconstruction_error:.6f}")
        except Exception as e:
            print(f"Final regularization attempt failed: {str(e)}")

            # Create emergency fallback basis with very high regularization
            basis = torch.eye(N, compressed_dim, dtype=torch.float32)
            pseudo_inverse = torch.eye(compressed_dim, N, dtype=torch.float32)

    # Verify shapes
    assert basis.shape == (N, compressed_dim), f"Basis shape should be ({N}, {compressed_dim}) but got {basis.shape}"
    assert pseudo_inverse.shape == (compressed_dim, N), f"Inverse shape should be ({compressed_dim}, {N}) but got {pseudo_inverse.shape}"

    return basis, pseudo_inverse

def estimate_signal_bandwidth(signal, threshold=0.95):
    """
    Estimate the bandwidth of a signal based on its frequency content

    Args:
        signal: Input signal (batch, N)
        threshold: Energy threshold (0-1)

    Returns:
        Estimated bandwidth in radians
    """
    # Convert to 1D if needed
    if signal.dim() > 1:
        # Take first signal in batch or average
        if signal.shape[0] > 1:
            # Average over batch dimension
            signal_1d = torch.mean(signal, dim=0)
        else:
            signal_1d = signal[0]
    else:
        signal_1d = signal

    # Compute FFT
    fft = torch.fft.rfft(signal_1d)
    magnitude = torch.abs(fft)

    # Calculate energy
    energy = torch.cumsum(magnitude**2, dim=0)

    # Normalize
    if energy[-1] > 0:
        energy = energy / energy[-1]

    # Find the bin where energy exceeds threshold
    threshold_bin = torch.where(energy > threshold)[0][0].item()

    # Convert bin to normalized bandwidth (0 to π)
    normalized_bandwidth = threshold_bin / len(signal_1d) * math.pi

    return normalized_bandwidth

def build_pswf_basis_adaptive(N, compressed_dim, signal=None, threshold=0.95,
                             default_bandwidth=math.pi, use_svd=True, eps=1e-10):
    """
    Build PSWF basis with adaptive bandwidth estimation

    Args:
        N: Original dimension
        compressed_dim: Target compressed dimension
        signal: Input signal for bandwidth estimation (optional)
        threshold: Energy threshold for bandwidth estimation (0-1)
        default_bandwidth: Default bandwidth if signal not provided
        use_svd: Whether to use SVD
        eps: Small value for numerical stability

    Returns:
        Tuple of (basis, pseudo_inverse)
    """
    # Estimate bandwidth if signal is provided
    if signal is not None:
        bandwidth = estimate_signal_bandwidth(signal, threshold)
        print(f"Estimated bandwidth from signal: {bandwidth:.4f} radians ({bandwidth/math.pi:.4f}π)")
    else:
        bandwidth = default_bandwidth
        print(f"Using default bandwidth: {bandwidth:.4f} radians ({bandwidth/math.pi:.4f}π)")

    # Build basis with estimated or default bandwidth
    return build_pswf_basis(N, compressed_dim, bandwidth=bandwidth, use_svd=use_svd, eps=eps)

# Make sure to define this first
def create_pswf_functions(input_dim, compressed_dim, signal=None, energy_threshold=0.95, bandwidth=math.pi,
                          per_vector=False, cache_basis=True):
    """
    Create PSWF projection functions with basis tailored to specific signals

    Args:
        input_dim: Original dimension
        compressed_dim: Target compressed dimension
        signal: Optional tensor to use for bandwidth estimation
               If shape is (batch, input_dim), can generate per-vector bases
        energy_threshold: Energy threshold for bandwidth estimation (0.0-1.0)
        bandwidth: Default bandwidth if no signal provided
        per_vector: If True and signal has multiple vectors, create a unique basis for each vector
        cache_basis: Whether to cache bases for efficiency (only relevant if per_vector=True)

    Returns:
        Tuple of (project_q, project_k, project_v, backproject) functions
    """
    # Dictionary to store bases if we're using per-vector approach with caching
    basis_cache = {}

    if signal is None:
        # Use default bandwidth and build a single basis
        print(f"Using default bandwidth: {bandwidth:.4f}")
        pswf_basis, pswf_inverse = build_pswf_basis(input_dim, compressed_dim, bandwidth)

        # Simple projection functions with fixed basis
        def pswf_project_q(x):
            return pswf_project(x, pswf_basis)

        def pswf_project_k(x):
            return pswf_project(x, pswf_basis)

        def pswf_project_v(x):
            return pswf_project(x, pswf_basis)

        def pswf_backproject_fn(x):
            return pswf_backproject(x, pswf_inverse)

    elif per_vector and signal.dim() > 1:
        # Create tailored basis for each vector in the batch
        print(f"Creating per-vector tailored PSWF bases for {signal.shape[0]} vectors")

        def pswf_project_q(x):
            # Handle batched input
            batch_size = x.shape[0]
            output = torch.zeros(batch_size, compressed_dim, device=x.device)

            for i in range(batch_size):
                # Generate or retrieve basis for this vector
                if cache_basis and i in basis_cache:
                    basis, _ = basis_cache[i]
                else:
                    # Estimate bandwidth for this specific vector
                    _, rel_bandwidth = estimate_bandwidth_bins(x[i], energy_threshold)
                    vector_bandwidth = math.pi * rel_bandwidth

                    # Build basis with tailored bandwidth
                    basis, inverse = build_pswf_basis(input_dim, compressed_dim, vector_bandwidth)

                    if cache_basis:
                        basis_cache[i] = (basis, inverse)

                # Project this vector
                output[i] = pswf_project(x[i].unsqueeze(0), basis).squeeze(0)

            return output

        # Similar implementations for K and V projections
        def pswf_project_k(x):
            # Similar to pswf_project_q
            batch_size = x.shape[0]
            output = torch.zeros(batch_size, compressed_dim, device=x.device)

            for i in range(batch_size):
                if cache_basis and i in basis_cache:
                    basis, _ = basis_cache[i]
                else:
                    _, rel_bandwidth = estimate_bandwidth_bins(x[i], energy_threshold)
                    vector_bandwidth = math.pi * rel_bandwidth
                    basis, inverse = build_pswf_basis(input_dim, compressed_dim, vector_bandwidth)

                    if cache_basis:
                        basis_cache[i] = (basis, inverse)

                output[i] = pswf_project(x[i].unsqueeze(0), basis).squeeze(0)

            return output

        def pswf_project_v(x):
            # Similar to pswf_project_q
            batch_size = x.shape[0]
            output = torch.zeros(batch_size, compressed_dim, device=x.device)

            for i in range(batch_size):
                if cache_basis and i in basis_cache:
                    basis, _ = basis_cache[i]
                else:
                    _, rel_bandwidth = estimate_bandwidth_bins(x[i], energy_threshold)
                    vector_bandwidth = math.pi * rel_bandwidth
                    basis, inverse = build_pswf_basis(input_dim, compressed_dim, vector_bandwidth)

                    if cache_basis:
                        basis_cache[i] = (basis, inverse)

                output[i] = pswf_project(x[i].unsqueeze(0), basis).squeeze(0)

            return output

        def pswf_backproject_fn(x):
            # Handle batched input for backprojection
            batch_size = x.shape[0]
            output = torch.zeros(batch_size, input_dim, device=x.device)

            for i in range(batch_size):
                if cache_basis and i in basis_cache:
                    _, inverse = basis_cache[i]
                else:
                    # If we somehow don't have the basis (shouldn't happen), use default
                    print("Warning: Missing basis for backprojection, using default")
                    _, inverse = build_pswf_basis(input_dim, compressed_dim, bandwidth)

                output[i] = pswf_backproject(x[i].unsqueeze(0), inverse).squeeze(0)

            return output

    else:
        # Use batch average for more stable estimation
        if signal.dim() > 1 and signal.shape[0] > 1:
            # Process multiple signals and average the bandwidth
            bandwidths = []
            for i in range(min(signal.shape[0], 10)):  # Limit to 10 samples for efficiency
                _, rel_band = estimate_bandwidth_bins(signal[i], energy_threshold)
                bandwidths.append(rel_band)

            avg_bandwidth = sum(bandwidths) / len(bandwidths)
            bandwidth = math.pi * avg_bandwidth
            print(f"Using signal-estimated average bandwidth: {bandwidth:.4f} (relative: {avg_bandwidth:.4f})")
        else:
            # Single signal
            _, rel_bandwidth = estimate_bandwidth_bins(signal.reshape(-1), energy_threshold)
            bandwidth = math.pi * rel_bandwidth
            print(f"Using signal-estimated bandwidth: {bandwidth:.4f} (relative: {rel_bandwidth:.4f})")

        # Build the basis with the determined bandwidth
        pswf_basis, pswf_inverse = build_pswf_basis(input_dim, compressed_dim, bandwidth)

        # Create the projection functions with shared basis
        def pswf_project_q(x):
            return pswf_project(x, pswf_basis)

        def pswf_project_k(x):
            return pswf_project(x, pswf_basis)

        def pswf_project_v(x):
            return pswf_project(x, pswf_basis)

        def pswf_backproject_fn(x):
            return pswf_backproject(x, pswf_inverse)

    return pswf_project_q, pswf_project_k, pswf_project_v, pswf_backproject_fn

def estimate_bandwidth_bins(signal, energy_threshold=0.95):
    """
    Estimate bandwidth of a 1D signal based on FFT energy

    Args:
        signal: 1D signal tensor
        energy_threshold: Fraction of energy to preserve (0.0-1.0)

    Returns:
        Tuple of (cutoff_index, relative_bandwidth)
    """
    # Validate input
    if signal.dim() > 1:
        signal = signal.reshape(-1)

    # Handle empty or zero signal
    if signal.numel() == 0 or torch.all(signal == 0):
        return 0, 0.0

    N = signal.shape[0]
    fft_signal = torch.fft.fft(signal)
    spectrum = torch.abs(fft_signal)[:N//2]

    # Handle numeric stability
    if torch.all(spectrum == 0):
        return 0, 0.0

    total_energy = torch.sum(spectrum**2)
    cumulative_energy = torch.cumsum(spectrum**2, dim=0)

    # Find first bin that exceeds threshold
    threshold_energy = energy_threshold * total_energy
    cutoff_indices = (cumulative_energy >= threshold_energy).nonzero()

    if cutoff_indices.numel() == 0:
        cutoff_idx = N // 2 - 1  # Use full bandwidth if threshold not met
    else:
        cutoff_idx = cutoff_indices[0].item()

    relative_bandwidth = (cutoff_idx + 1) / (N/2)  # +1 to avoid zero bandwidth

    return cutoff_idx, relative_bandwidth

def pswf_op(N, dtype=torch.float32, device=None):

    if device is None:
        device = torch.device('cpu')
    if N <= 0:
        raise ValueError("N must be a positive integer.")

    cutoff_bin, rel_bandwidth = estimate_bandwidth_bins(signal)
    bandwidth = math.pi * rel_bandwidth
    print(f"Estimated relative bandwidth: {rel_bandwidth:.2f}, c = {bandwidth:.2f}")

    # Discretize [-1,1]
    x = torch.linspace(-1, 1, N, device=device, dtype=dtype)
    dx = x[1] - x[0]

    # Build the sinc kernel matrix
    X, Y = torch.meshgrid(x, x, indexing='ij')
    kernel = torch.sinc(bandwidth * (X - Y) / math.pi) * dx

    # Solve symmetric eigenproblem
    eigenvalues, eigenvectors = torch.linalg.eigh(kernel)

    # Sort eigenvalues and eigenvectors descending
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return x, eigenvectors, eigenvalues

def pswf_transform(signal, pswf_matrix):

    return pswf_matrix.T @ signal

def pswf_inverse(coeffs, pswf_matrix):

    return pswf_matrix @ coeffs

def pswf_transform_2d(X, pswf_row, pswf_col):
    temp = pswf_row.T @ X
    transformed = temp @ pswf_col
    return transformed

def pswf_inverse_2d(X_transformed, pswf_row, pswf_col):
    temp = pswf_row @ X_transformed
    reconstructed = temp @ pswf_col.T
    return reconstructed

def estimate_bandwidth_2d(matrix, energy_threshold=0.95):
    """
    Estimate row-wise and column-wise bandwidths for a 2D matrix.

    Args:
        matrix (Tensor): 2D tensor (N, M)

    Returns:
        c_row: Bandlimit c for rows
        c_col: Bandlimit c for columns
    """
    N, M = matrix.shape

    # Row-wise estimation
    rel_bandwidth_rows = []
    for i in range(N):
        _, rel_band = estimate_bandwidth_bins(matrix[i, :])
        rel_bandwidth_rows.append(rel_band)
    avg_rel_bandwidth_row = sum(rel_bandwidth_rows) / len(rel_bandwidth_rows)
    c_row = math.pi * avg_rel_bandwidth_row

    # Column-wise estimation
    rel_bandwidth_cols = []
    for j in range(M):
        _, rel_band = estimate_bandwidth_bins(matrix[:, j])
        rel_bandwidth_cols.append(rel_band)
    avg_rel_bandwidth_col = sum(rel_bandwidth_cols) / len(rel_bandwidth_cols)
    c_col = math.pi * avg_rel_bandwidth_col

    return c_row, c_col

def create_attention_head_specific_pswf_functions(model, tokenizer, prompts, device,
                                                 input_dim, compressed_dim, layer_idx=0, head_idx=0):
    """
    Create PSWF functions with bandwidth tailored to specific attention head

    Args:
        model: The Transformer model
        tokenizer: Tokenizer for processing prompts
        prompts: List of text prompts to generate activations
        device: Device to run on
        input_dim: Dimension of input vectors
        compressed_dim: Target compressed dimension
        layer_idx: Index of the transformer layer to analyze
        head_idx: Index of the attention head to analyze

    Returns:
        Tuple of (project_q, project_k, project_v, backproject) functions
    """
    # Get activations for the specific head
    q_activations, k_activations, v_activations = get_head_specific_activations(
        model, tokenizer, prompts, device, layer_idx, head_idx)

    # Create projection functions for queries
    q_project_fn, _, _, _ = create_pswf_functions(
        input_dim, compressed_dim, signal=q_activations, per_vector=False)

    # Create projection functions for keys
    _, k_project_fn, _, _ = create_pswf_functions(
        input_dim, compressed_dim, signal=k_activations, per_vector=False)

    # Create projection functions for values
    _, _, v_project_fn, _ = create_pswf_functions(
        input_dim, compressed_dim, signal=v_activations, per_vector=False)

    # Create backprojection function (using value activations which are most important)
    _, _, _, backproject_fn = create_pswf_functions(
        input_dim, compressed_dim, signal=v_activations, per_vector=False)

    return q_project_fn, k_project_fn, v_project_fn, backproject_fn

def get_head_specific_activations(model, tokenizer, prompts, device, layer_idx=0, head_idx=0):
    """
    Get query, key, value activations for a specific attention head

    Returns:
        Tuple of (q_activations, k_activations, v_activations)
    """
    # Container for activations
    q_activations = []
    k_activations = []
    v_activations = []

    # Define hook function to capture activations
    def hook_fn(module, input, output):
        # Get query, key, value projections
        # This assumes GPT-2 style model with .attn module containing q, k, v projections
        hidden_states = input[0]
        batch_size = hidden_states.shape[0]
        q = module.q_proj(hidden_states)
        k = module.k_proj(hidden_states)
        v = module.v_proj(hidden_states)

        # Reshape to get per-head activations
        head_dim = q.shape[-1] // module.num_heads

        q = q.view(batch_size, -1, module.num_heads, head_dim)
        k = k.view(batch_size, -1, module.num_heads, head_dim)
        v = v.view(batch_size, -1, module.num_heads, head_dim)

        # Extract specific head activations and add to our lists
        q_activations.append(q[:, :, head_idx, :].reshape(-1, head_dim).detach())
        k_activations.append(k[:, :, head_idx, :].reshape(-1, head_dim).detach())
        v_activations.append(v[:, :, head_idx, :].reshape(-1, head_dim).detach())

    # Register hook on specific layer
    hook_handle = model.transformer.h[layer_idx].attn.register_forward_hook(hook_fn)

    # Process prompts
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        _ = model(**inputs)

    # Remove the hook
    hook_handle.remove()

    # Concatenate all activations
    q_all = torch.cat(q_activations, dim=0)
    k_all = torch.cat(k_activations, dim=0)
    v_all = torch.cat(v_activations, dim=0)

    return q_all, k_all, v_all

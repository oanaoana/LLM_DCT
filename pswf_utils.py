import torch
import math

def pswf_project(x, pswf_basis):
    """Project input onto PSWF basis"""
    if pswf_basis.device != x.device:
        pswf_basis = pswf_basis.to(x.device)

    # Project onto basis
    return torch.matmul(x, pswf_basis)

def pswf_backproject(x, pswf_inverse):
    """
    Project back from compressed space to original space
    x: (batch, compressed_dim)
    pswf_inverse: (compressed_dim, dim) - should be transposed if needed
    """
    # Check if dimensions are correct
    if pswf_inverse.shape[0] != x.shape[1]:
        # Transpose the inverse matrix if it's in the wrong orientation
        pswf_inverse = pswf_inverse.T

    # Ensure x is on the correct device
    if pswf_inverse.device != x.device:
        pswf_inverse = pswf_inverse.to(x.device)

    # Back-project via matrix multiplication
    return torch.matmul(x, pswf_inverse)

def build_pswf_basis(N, compressed_dim, bandwidth=math.pi):
    """Build PSWF basis for projection and back-projection"""
    # Input validation
    assert compressed_dim <= N, f"Compressed dimension {compressed_dim} cannot exceed original dimension {N}"
    if compressed_dim == N:
        # Return identity matrices for no compression
        return torch.eye(N), torch.eye(N)

    # Create discretization grid
    x = torch.linspace(-1, 1, N)
    dx = x[1] - x[0]

    # Build kernel matrix
    X, Y = torch.meshgrid(x, x, indexing='ij')
    kernel = torch.sinc(bandwidth * (X - Y) / math.pi) * dx

    # Compute eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(kernel)

    # Sort by decreasing eigenvalue
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select top eigenvectors for basis
    basis = eigenvectors[:, :compressed_dim]  # [N, compressed_dim]

    # Compute pseudo-inverse for back-projection
    # The pseudo-inverse should be [compressed_dim, N] for backprojection
    pseudo_inverse = torch.linalg.pinv(basis)  # This will give [compressed_dim, N]

    return basis, pseudo_inverse

# Make sure to define this first
def create_pswf_functions(input_dim, compressed_dim, bandwidth=math.pi):
    """Create PSWF projection functions with proper basis"""
    # Build the basis
    pswf_basis, pswf_inverse = build_pswf_basis(input_dim, compressed_dim, bandwidth)

    # Create the projection functions
    def pswf_project_q(x):
        return pswf_project(x, pswf_basis)

    def pswf_project_k(x):
        return pswf_project(x, pswf_basis)

    def pswf_project_v(x):
        return pswf_project(x, pswf_basis)

    def pswf_backproject(x):
        return pswf_project(x, pswf_inverse)

    return pswf_project_q, pswf_project_k, pswf_project_v, pswf_backproject

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

def estimate_bandwidth_bins(signal, energy_threshold=0.95):
    """
    Estimate bandwidth of a 1D signal based purely on FFT bins.
    """
    N = signal.shape[0]
    fft_signal = torch.fft.fft(signal)
    spectrum = torch.abs(fft_signal)[:N//2]
    total_energy = torch.sum(spectrum**2)
    cumulative_energy = torch.cumsum(spectrum**2, dim=0)

    cutoff_idx = (cumulative_energy >= energy_threshold * total_energy).nonzero()[0].item()
    relative_bandwidth = cutoff_idx / (N/2)

    return cutoff_idx, relative_bandwidth

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

import torch
import math
import numpy as np
import time
import config
import scipy.fft as fft

def build_dct_basis(input_dim, compressed_dim):
    """
    Build DCT basis for dimensionality reduction

    Args:
        input_dim: Original dimension
        compressed_dim: Target compressed dimension

    Returns:
        Tuple of (basis, inverse)
    """
    # Input validation
    assert compressed_dim <= input_dim, f"Compressed dimension {compressed_dim} cannot exceed input dimension {input_dim}"

    # For no compression case, return identity matrices
    if compressed_dim == input_dim:
        return torch.eye(input_dim), torch.eye(input_dim)

    # Build the DCT matrix
    dct_mat = torch.zeros(input_dim, input_dim)
    for k in range(input_dim):
        for n in range(input_dim):
            dct_mat[k, n] = torch.sqrt(torch.tensor(2.0 / input_dim)) * \
                            torch.cos(torch.tensor((2*n + 1) * k * math.pi / (2 * input_dim)))
            if k == 0:
                dct_mat[k, n] *= math.sqrt(2) / 2

    # Take only the first compressed_dim rows for the basis
    basis = dct_mat[:compressed_dim, :].T  # Transpose to get (input_dim, compressed_dim)

    # Create pseudoinverse for back-projection (compressed_dim, input_dim)
    inverse = dct_mat[:compressed_dim, :]

    # Verify shapes before returning
    assert basis.shape == (input_dim, compressed_dim), f"DCT basis shape wrong: {basis.shape}"
    assert inverse.shape == (compressed_dim, input_dim), f"DCT inverse shape wrong: {inverse.shape}"

    return basis, inverse

def construct_dct_basis(input_dim, output_dim):
    """Build DCT basis for compression"""
    # Create basis matrix
    basis = np.zeros((input_dim, output_dim))

    # Fill with DCT basis vectors
    for k in range(output_dim):
        if k == 0:
            basis[:, k] = 1.0 / math.sqrt(input_dim)
        else:
            for n in range(input_dim):
                basis[n, k] = math.sqrt(2.0 / input_dim) * math.cos(
                    (math.pi * (2 * n + 1) * k) / (2 * input_dim)
                )


    # Convert to torch tensor
    basis = torch.from_numpy(basis.astype(np.float32))

    # DCT basis is orthogonal, so transpose is inverse
    inverse = basis.T

    return basis, inverse

def dct_project(x, dct_basis):
    """
    Project input onto DCT basis to get compressed representation

    Args:
        x: Input tensor of shape (batch, input_dim)
        dct_basis: DCT basis matrix of shape (input_dim, compressed_dim)

    Returns:
        Compressed tensor of shape (batch, compressed_dim)
    """
    # Check input shape
    input_dim = dct_basis.shape[0]
    compressed_dim = dct_basis.shape[1]
    assert x.shape[1] == input_dim, f"Input dimension mismatch: {x.shape[1]} vs {input_dim}"

    # Ensure x is on the correct device
    if dct_basis.device != x.device:
        dct_basis = dct_basis.to(x.device)

    # Project onto basis
    return torch.matmul(x, dct_basis)

def dct_backproject(x, dct_inverse):
    """
    Project back from compressed representation to original space

    Args:
        x: Compressed tensor of shape (batch, compressed_dim)
        dct_inverse: Inverse DCT basis of shape (compressed_dim, input_dim)

    Returns:
        Reconstructed tensor of shape (batch, input_dim)
    """
    # Check shape
    compressed_dim = x.shape[1]
    assert dct_inverse.shape[0] == compressed_dim, f"Dimension mismatch: {dct_inverse.shape[0]} vs {compressed_dim}"

    # Ensure x is on the correct device
    if dct_inverse.device != x.device:
        dct_inverse = dct_inverse.to(x.device)

    # Back-project via matrix multiplication
    return torch.matmul(x, dct_inverse)

def get_available_device():
    """
    Returns the best available device for computation:
    CUDA GPU, ROCm GPU, MPS, or CPU as last resort.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch, 'hip') and torch.hip.is_available():  # Check for ROCm/AMD
        return torch.device('hip')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def dct_op(N, dtype=torch.float32, device=None):

    if device is None:
        device = get_available_device()
    if N <= 0:
        raise ValueError("N must be a positive integer.")
    matrix = torch.zeros((N, N), dtype=dtype, device=device)
    for i in range(N):
        for j in range(N):
            matrix[i, j] = math.sqrt(2 / N) * math.cos(math.pi * i * (2 * j + 1) / (2 * N))
            if i == 0:
                matrix[i, j] *= math.sqrt(2)/2
    return matrix

def idct_op(N, dtype=torch.float32, device=None):
    if device is None:
        device = get_available_device()
    dct_matrix = dct_op(N, dtype, device)
    dct_matrix = dct_matrix.T
    ##dct_matrix[0, :] *= math.sqrt(2)
    return dct_matrix

def dct_vii_op(N, dtype=torch.float32, device=None):
    if device is None:
        device = get_available_device()
    if N <= 0:
        raise ValueError("N must be a positive integer.")

    matrix = torch.zeros((N, N), dtype=dtype, device=device)
    for i in range(N):
        for j in range(N):
            theta = math.pi * (i + 0.25) * (j + 0.25) / N
            matrix[i, j] = math.sqrt(2 / N) * math.cos(theta)
    return matrix

def dct_viii_op(N, dtype=torch.float32, device=None):
    if device is None:
        device = get_available_device()
    if N <= 0:
        raise ValueError("N must be a positive integer.")

    matrix = torch.zeros((N, N), dtype=dtype, device=device)
    for i in range(N):
        for j in range(N):
            theta = math.pi * (i + 0.75) * (j + 0.75) / N
            matrix[i, j] = math.sqrt(2 / N) * math.cos(theta)
    return matrix

def mdct_op(N, dtype=torch.float32, device=None):
    if device is None:
        device = get_available_device()
    if N <= 0:
        raise ValueError("N must be a positive integer.")

    matrix = torch.zeros((N, 2 * N), dtype=dtype, device=device)
    for k in range(N):
        for n in range(2 * N):
            angle = math.pi / N * (n + 0.5 + N / 2) * (k + 0.5)
            matrix[k, n] = math.sqrt(1 / N) * math.cos(angle)
    return matrix

# Add a cache for DCT matrices
dct_matrix_cache = {}
idct_matrix_cache = {}

def get_dct_matrix(config):
    """
    Get DCT matrix from cache or compute it.

    Args:
        N (int): Size of the DCT matrix.
        dtype (torch.dtype): Data type of the matrix.
        device (torch.device): Device to store the matrix.
        dct_type (str): Type of DCT to use ('default', 'vii', 'viii').

    Returns:
        torch.Tensor: The DCT matrix.
    """
    N = config.block_size
    dtype = config.dtype if hasattr(config, 'dtype') else torch.float32
    device = config.device if hasattr(config, 'device') else get_available_device()
    dct_type = config.dct_type

    key = (N, dtype, str(device), dct_type)
    if key not in dct_matrix_cache:
        if dct_type == 'default':
            dct_matrix_cache[key] = dct_op(N, dtype, device)
        elif dct_type == 'vii':
            dct_matrix_cache[key] = dct_vii_op(N, dtype, device)
        elif dct_type == 'viii':
            dct_matrix_cache[key] = dct_viii_op(N, dtype, device)
        else:
            raise ValueError(f"Unsupported DCT type: {dct_type}")
    return dct_matrix_cache[key]

def get_idct_matrix(config):
    """Get IDCT matrix from cache or compute it."""
    N = config.block_size
    dtype = config.dtype if hasattr(config, 'dtype') else torch.float32
    device = config.device if hasattr(config, 'device') else get_available_device()
    dct_type = config.dct_type

    key = (N, dtype, str(device), dct_type)
    if key not in idct_matrix_cache:
        if dct_type == 'default':
            idct_matrix_cache[key] = idct_op(N, dtype, device)
        elif dct_type == 'vii':
            idct_matrix_cache[key] = dct_vii_op(N, dtype, device).T
        elif dct_type == 'viii':
            idct_matrix_cache[key] = dct_viii_op(N, dtype, device).T
        else:
            raise ValueError(f"Unsupported DCT type: {dct_type}")
    return idct_matrix_cache[key]

def apply_operator_dct(x, config, direction='forward'):
    """
    Efficiently apply DCT or IDCT using the matrix operator approach.
    Computes dct_matrix * x * dct_matrix.T for 2D matrices or dct_matrix * x for 1D vectors.

    Args:
        x (torch.Tensor): Input tensor (1D vector or 2D matrix).
        direction (str): 'forward' for DCT, 'inverse' for IDCT.
        dct_type (str): Type of DCT to use ('default', 'vii', 'viii').

    Returns:
        torch.Tensor: Transformed data (DCT or IDCT).
    """
    if config is None:
        raise ValueError("A valid configuration object (config) must be provided.")
    if direction not in ['forward', 'inverse']:
        raise ValueError("Invalid direction. Use 'forward' for DCT or 'inverse' for IDCT.")
    if config.dct_type not in ['default', 'vii', 'viii']:
        raise ValueError("Invalid DCT type. Use 'default', 'vii', or 'viii'.")

    # Choose the appropriate matrix based on the direction and DCT type
    if direction == 'forward':
        transform_matrix = get_dct_matrix(config)
    else:  # direction == 'inverse'
        transform_matrix = get_idct_matrix(config)

    if len(x.shape) == 2:
        # 2D case: Apply transform to rows and columns
        return torch.matmul(transform_matrix, torch.matmul(x, transform_matrix.T))
    elif len(x.shape) == 1:
        # 1D case: Apply transform to the vector
        return torch.matmul(transform_matrix, x)
    else:
        raise ValueError("Input must be a 1D vector or 2D matrix.")

def truncate_operator_dct(x, config):
    """
    Truncate the input in DCT space by retaining only the first `max_freqs` frequencies.

    Args:
        x (torch.Tensor): Input tensor (1D vector or 2D matrix).
        max_freqs (int): Number of frequencies to retain.
        method (str): Method to use for DCT ('operator' or 'fft').
        norm (str): Normalization method ('ortho' for orthonormal).

    Returns:
        torch.Tensor: Truncated input in the original domain.
    """
    max_freqs = config.max_freqs
    dct_type = config.dct_type

    if max_freqs <= 0 or max_freqs > x.flatten().size(-1):
        raise ValueError(f"max_freqs must be in the range [1, {x.size(-1)}], got {max_freqs}.")

    # Apply DCT to the input
    x_dct = apply_operator_dct(x, config, direction='forward')

    # Truncate the DCT coefficients
    if len(x.shape) == 2:  # 2D case
        truncated_dct = torch.zeros_like(x_dct)
        x_dct_abs = torch.abs(x_dct)
        sorted_indices = torch.argsort(x_dct_abs.flatten(), descending=True)
        truncated_dct_flat = x_dct.flatten()
        truncated_dct_flat[sorted_indices[max_freqs:]] = 0
        truncated_dct = truncated_dct_flat.reshape(x_dct.shape)
        #print(f"Transformed DCT vector: {x_dct.round(decimals=4)}")
        #print(f"Truncated DCT vector: {truncated_dct.round(decimals=4)}")
    elif len(x.shape) == 1:  # 1D case
        truncated_dct = torch.zeros_like(x_dct)
        x_dct_sort, indices = torch.sort(torch.abs(x_dct), descending=True)
        truncated_dct = x_dct.clone()
        truncated_dct[indices[max_freqs:]] = 0
        #truncated_dct[:max_freqs] = x_dct[:max_freqs]
        #print(f"Transformed DCT vector: {x_dct}")
        #print(f"Truncated DCT vector: {truncated_dct}")
    else:
        raise ValueError("Input must be a 1D vector or 2D matrix.")

    # Apply inverse DCT to reconstruct the truncated input
    x_truncated = apply_operator_dct(truncated_dct, config, direction='inverse')

    return x_truncated

def truncate_fft_dct(x, max_freqs, norm='ortho'):
    """
    Truncate the input in DCT space using the FFT-based DCT method by retaining only the first `max_freqs` frequencies.

    Args:
        x (torch.Tensor): Input tensor (1D vector or 2D matrix).
        max_freqs (int): Number of frequencies to retain.
        norm (str): Normalization method ('ortho' for orthonormal).

    Returns:
        torch.Tensor: Truncated input in the original domain.
    """
    if max_freqs <= 0 or max_freqs > x.size(-1):
        raise ValueError(f"max_freqs must be in the range [1, {x.size(-1)}], got {max_freqs}.")

    # Apply FFT-based DCT to the input
    x_dct = dct_fft(x, norm=norm)

    # Truncate the DCT coefficients
    if len(x.shape) == 2:  # 2D case
        truncated_dct = torch.zeros_like(x_dct)
        truncated_dct[:max_freqs, :max_freqs] = x_dct[:max_freqs, :max_freqs]
    elif len(x.shape) == 1:  # 1D case
        truncated_dct = torch.zeros_like(x_dct)
        truncated_dct[:max_freqs] = x_dct[:max_freqs]
    else:
        raise ValueError("Input must be a 1D vector or 2D matrix.")

    # Apply inverse FFT-based DCT to reconstruct the truncated input
    x_truncated = idct_fft(truncated_dct, norm=norm)

    return x_truncated

def truncate_fft_dct_sort(x, max_freqs, norm='ortho'):
    """
    Truncate the input in DCT space using the FFT-based DCT method by retaining only the first `max_freqs` frequencies.

    Args:
        x (torch.Tensor): Input tensor (1D vector or 2D matrix).
        max_freqs (int): Number of frequencies to retain.
        norm (str): Normalization method ('ortho' for orthonormal).

    Returns:
        torch.Tensor: Truncated input in the original domain.
    """
    if max_freqs <= 0 or max_freqs > x.flatten().size(-1):
        raise ValueError(f"max_freqs must be in the range [1, {x.size(-1)}], got {max_freqs}.")

    # Apply FFT-based DCT to the input
    x_dct = dct_fft(x, norm=norm)

    # Truncate the DCT coefficients
    if len(x.shape) == 2:  # 2D case
        truncated_dct = torch.zeros_like(x_dct)
        x_dct_abs = torch.abs(x_dct)
        sorted_indices = torch.argsort(x_dct_abs.flatten(), descending=True)
        truncated_dct_flat = x_dct.flatten()
        truncated_dct_flat[sorted_indices[max_freqs:]] = 0
        truncated_dct = truncated_dct_flat.reshape(x_dct.shape)
        #print(f"Transformed DCT vector (FFT): {x_dct.round(decimals=4)}")
        #print(f"Truncated DCT vector (FFT): {truncated_dct.round(decimals=4)}")
    elif len(x.shape) == 1:  # 1D case
        truncated_dct = torch.zeros_like(x_dct)
        x_dct_sort, indices = torch.sort(torch.abs(x_dct), descending=True)
        truncated_dct = x_dct.clone()
        truncated_dct[indices[max_freqs:]] = 0
        #print(f"Transformed DCT vector (FFT): {x_dct}")
        #print(f"Truncated DCT vector (FFT): {truncated_dct}")
    else:
        raise ValueError("Input must be a 1D vector or 2D matrix.")

    # Apply inverse FFT-based DCT to reconstruct the truncated input
    x_truncated = idct_fft(truncated_dct, norm=norm)

    return x_truncated

def dct_fft(src, dim=-1, norm='ortho'):
    # type: (torch.tensor, int, str) -> torch.tensor

    x = src.clone()
    N = x.shape[dim]

    x = x.transpose(dim, -1)
    x_shape = x.shape
    x = x.contiguous().view(-1, N)

    v = torch.empty_like(x, device=x.device)
    v[..., :(N - 1) // 2 + 1] = x[..., ::2]

    if N % 2:  # odd length
        v[..., (N - 1) // 2 + 1:] = x.flip(-1)[..., 1::2]
    else:  # even length
        v[..., (N - 1) // 2 + 1:] = x.flip(-1)[..., ::2]

    V = torch.fft.fft(v, dim=-1)

    k = torch.arange(N, device=x.device)
    V = 2 * V * torch.exp(-1j * np.pi * k / (2 * N))

    if norm == 'ortho':
        V[..., 0] *= math.sqrt(1/(4*N))
        V[..., 1:] *= math.sqrt(1/(2*N))

    V = V.real
    V = V.view(*x_shape).transpose(-1, dim)

    return V

def idct_fft(src, dim=-1, norm='ortho'):
    # type: (torch.tensor, int, str) -> torch.tensor

    X = src.clone()
    N = X.shape[dim]

    X = X.transpose(dim, -1)
    X_shape = X.shape
    X = X.contiguous().view(-1, N)

    if norm == 'ortho':
        X[..., 0] *= 1 / math.sqrt(2)
        X *= N*math.sqrt((2 / N))
    else:
        raise Exception("idct with norm=None is buggy A.F")

    k = torch.arange(N, device=X.device)

    X = X * torch.exp(1j * np.pi * k / (2 * N))
    X = torch.fft.ifft(X, dim=-1).real
    v = torch.empty_like(X, device=X.device)

    v[..., ::2] = X[..., :(N - 1) // 2 + 1]
    v[..., 1::2] = X[..., (N - 1) // 2 + 1:].flip(-1)

    v = v.view(*X_shape).transpose(-1, dim)

    return v

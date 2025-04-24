import torch
import math
import time

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
    #matrix = torch.zeros((N, N), dtype=dtype, device=device)
    #for i in range(N):
    #    for j in range(N):
    #        matrix[j, i] = math.sqrt(2 / N) * math.cos(math.pi * i * (2 * j + 1) / (2 * N))
    #        if i == 0:
    #            matrix[j, i] *= math.sqrt(2)/2
    #return matrix
    dct_matrix = dct_op(N, dtype, device)
    dct_matrix = dct_matrix.T
    ##dct_matrix[0, :] *= math.sqrt(2)
    return dct_matrix

# Add a cache for DCT matrices
dct_matrix_cache = {}
idct_matrix_cache = {}

def get_dct_matrix(N, dtype=torch.float32, device=None):
    """Get DCT matrix from cache or compute it."""
    if device is None:
        device = get_available_device()

    key = (N, dtype, str(device))
    if key not in dct_matrix_cache:
        dct_matrix_cache[key] = dct_op(N, dtype, device)
    return dct_matrix_cache[key]

def get_idct_matrix(N, dtype=torch.float32, device=None):
    """Get IDCT matrix from cache or compute it."""
    if device is None:
        device = get_available_device()

    key = (N, dtype, str(device))
    if key not in idct_matrix_cache:
        idct_matrix_cache[key] = idct_op(N, dtype, device)
    return idct_matrix_cache[key]

def apply_dct(x, method='operator', norm='ortho'):
    """
    Apply Discrete Cosine Transform to input data using specified method.

    Args:
        x (torch.Tensor): Input tensor (vector or matrix)
        method (str): Method to use: 'operator' or 'fft'
        norm (str): Normalization method ('ortho' for orthonormal)

    Returns:
        torch.Tensor: DCT transformed data
    """
    orig_shape = x.shape

    # For matrices or higher-dimensional tensors, apply DCT to the last dimension
    if len(orig_shape) > 1:
        # Reshape to treat last dimension separately
        x_reshaped = x.reshape(-1, orig_shape[-1])
        result = []

        # Process each vector along the last dimension
        for i in range(x_reshaped.shape[0]):
            vector = x_reshaped[i]
            # Apply the selected DCT method
            if method == 'operator':
                dct_result = apply_operator_dct(vector, norm)
            elif method == 'fft':
                dct_result = dct_fft(vector, norm)
            else:
                raise ValueError(f"Unknown method: {method}. Use 'operator' or 'fft'.")

            result.append(dct_result)

        # Combine results and reshape back to original dimensions
        return torch.stack(result).reshape(orig_shape)
    else:
        # Vector case
        if method == 'operator':
            return apply_operator_dct(x, norm)
        elif method == 'fft':
            return dct_fft(x, norm)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'operator' or 'fft'.")

def apply_operator_dct_efficient(x, direction='forward'):
    """
    Efficiently apply DCT or IDCT using the matrix operator approach.
    Computes dct_matrix * x * dct_matrix.T for 2D matrices or dct_matrix * x for 1D vectors.

    Args:
        x (torch.Tensor): Input tensor (1D vector or 2D matrix).
        norm (str): Normalization method ('ortho' for orthonormal).
        direction (str): 'forward' for DCT, 'inverse' for IDCT.

    Returns:
        torch.Tensor: Transformed data (DCT or IDCT).
    """
    if direction not in ['forward', 'inverse']:
        raise ValueError("Invalid direction. Use 'forward' for DCT or 'inverse' for IDCT.")

    # Choose the appropriate matrix based on the direction
    if direction == 'forward':
        transform_matrix = get_dct_matrix(x.size(-1), dtype=x.dtype, device=x.device)
    else:  # direction == 'inverse'
        transform_matrix = get_idct_matrix(x.size(-1), dtype=x.dtype, device=x.device)

    if len(x.shape) == 2:
        # 2D case: Apply transform to rows and columns
        return torch.matmul(transform_matrix, torch.matmul(x, transform_matrix.T))
    elif len(x.shape) == 1:
        # 1D case: Apply transform to the vector
        return torch.matmul(transform_matrix, x)
    else:
        raise ValueError("Input must be a 1D vector or 2D matrix.")

def truncate_operator_dct(x, max_freqs):
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
    if max_freqs <= 0 or max_freqs > x.flatten().size(-1):
        raise ValueError(f"max_freqs must be in the range [1, {x.size(-1)}], got {max_freqs}.")

    # Apply DCT to the input
    x_dct = apply_operator_dct_efficient(x, direction='forward')

    # Truncate the DCT coefficients
    if len(x.shape) == 2:  # 2D case
        truncated_dct = torch.zeros_like(x_dct)
        x_dct_abs = torch.abs(x_dct)
        sorted_indices = torch.argsort(x_dct_abs.flatten(), descending=True)
        truncated_dct_flat = x_dct.flatten()
        truncated_dct_flat[sorted_indices[max_freqs:]] = 0
        truncated_dct = truncated_dct_flat.reshape(x_dct.shape)
        print(f"Transformed DCT vector: {x_dct.round(decimals=4)}")
        print(f"Truncated DCT vector: {truncated_dct.round(decimals=4)}")
    elif len(x.shape) == 1:  # 1D case
        truncated_dct = torch.zeros_like(x_dct)
        x_dct_sort, indices = torch.sort(torch.abs(x_dct), descending=True)
        truncated_dct = x_dct.clone()
        truncated_dct[indices[max_freqs:]] = 0
        #truncated_dct[:max_freqs] = x_dct[:max_freqs]
        print(f"Transformed DCT vector: {x_dct}")
        print(f"Truncated DCT vector: {truncated_dct}")
    else:
        raise ValueError("Input must be a 1D vector or 2D matrix.")

    # Apply inverse DCT to reconstruct the truncated input
    x_truncated = apply_operator_dct_efficient(truncated_dct, direction='inverse')

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

def apply_idct(x, method='operator', norm='ortho'):
    """
    Apply Inverse Discrete Cosine Transform using specified method.

    Args:
        x (torch.Tensor): Input tensor in DCT domain
        method (str): Method to use: 'operator' or 'fft'
        norm (str): Normalization method

    Returns:
        torch.Tensor: Reconstructed data in spatial domain
    """
    orig_shape = x.shape

    # For matrices or higher-dimensional tensors, apply IDCT to the last dimension
    if len(orig_shape) > 1:
        # Reshape to treat last dimension separately
        x_reshaped = x.reshape(-1, orig_shape[-1])
        result = []

        # Process each vector along the last dimension
        for i in range(x_reshaped.shape[0]):
            vector = x_reshaped[i]
            # Apply the selected IDCT method
            if method == 'operator':
                idct_result = apply_operator_idct(vector, norm)
            elif method == 'fft':
                idct_result = idct_fft(vector, norm)
            else:
                raise ValueError(f"Unknown method: {method}. Use 'operator' or 'fft'.")

            result.append(idct_result)

        # Combine results and reshape back to original dimensions
        return torch.stack(result).reshape(orig_shape)
    else:
        # Vector case
        if method == 'operator':
            return apply_operator_idct(x, norm)
        elif method == 'fft':
            return idct_fft(x, norm)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'operator' or 'fft'.")

def apply_operator_idct(x, norm='ortho'):
    """
    Apply IDCT using the matrix operator approach.

    Args:
        x (torch.Tensor): Input vector in DCT domain
        norm (str): Normalization method

    Returns:
        torch.Tensor: Reconstructed vector in spatial domain
    """
    # Create IDCT matrix
    idct_matrix = idct_op(x.size(-1), dtype=x.dtype, device=x.device)

    # Apply transformation
    return torch.matmul(idct_matrix, x)

def dct_fft(x, norm='ortho'):
    orig_dtype = x.dtype
    orig_device = x.device
    x = x.to(dtype=torch.float32, device=orig_device)

    N = x.size(-1)
    v = torch.cat([x, x.flip(dims=[-1])], dim=-1)
    Vc = torch.fft.fft(v, dim=-1)
    k = torch.arange(N, device=orig_device)
    factor = -1j * math.pi * k / (2 * N)
    W = torch.exp(factor).unsqueeze(0).to(orig_device)
    X = (Vc[..., :N] * W).real

    if norm == 'ortho':
        X[..., 0] = X[..., 0] * math.sqrt(2)/2
        X = X * math.sqrt(2 /  N)
    else:
        X = X / 2

    return X.to(dtype=orig_dtype)

def idct_fft(X, norm='ortho'):
    orig_dtype = X.dtype
    orig_device = X.device
    X = X.to(dtype=torch.float32, device=orig_device)

    N = X.size(-1)
    X0 = X[..., 0].unsqueeze(-1)

    if norm == 'ortho':
        X0 = X0 * math.sqrt(2)/2
        X = X * math.sqrt(2 / N)

    X = torch.cat([X0, X[..., 1:]], dim=-1)
    V = torch.zeros(X.size(0), 2 * N, device=orig_device, dtype=torch.complex64)
    k = torch.arange(N, device=orig_device)
    factor = 1j * math.pi * k / (2 * N)
    W = torch.exp(factor).unsqueeze(0).to(orig_device)
    V[..., :N] = X * W
    flipped_conj = torch.conj(V[..., 1:N].flip(dims=[-1]))
    V[..., -N + 1:] = flipped_conj

    v = torch.fft.ifft(V, dim=-1).real
    return v[..., :N].to(dtype=orig_dtype)

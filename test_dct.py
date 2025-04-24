import torch
import math
import sys
import os
from dct_utils import get_dct_matrix, get_idct_matrix, truncate_operator_dct, truncate_fft_dct, truncate_fft_dct_sort

N = 6
# Generate a DCT matrix
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
dct_matrix = get_dct_matrix(N, device=device, dtype=dtype)
idct_matrix = get_idct_matrix(N, device=device, dtype=dtype)
print("DCT matrix:")
#print(dct_matrix.round(decimals=4))

# Test 1D truncation
x = torch.cos(torch.randn(16))
max_freqs = 8
x_truncated = truncate_operator_dct(x, max_freqs=max_freqs)
error = torch.norm(x - x_truncated)
print(f"Difference: {torch.abs(x-x_truncated).round(decimals=4)}")
print(f"Reconstruction error (1D): {error.item()}")

#sys.exit()
# Test 2D truncation
x = torch.randn(16, 16)
# Save the x vector to a file for later use
# Save the x vector to a file for later use
file_path = os.path.join(os.path.dirname(__file__), "x_vector.pt")
#torch.save(x, file_path)

# Load the x vector from the file
x = torch.load(file_path)

max_freqs = 144
x_truncated = truncate_operator_dct(x, max_freqs=max_freqs)
error = torch.norm(x - x_truncated, p=2)
#print(f"Difference: {torch.abs(x-x_truncated).round(decimals=4)}")
print(f"Reconstruction error (2D): {error.item()}")

# Load the x vector from the file
x = torch.load(file_path)

max_freqs = 12
x_truncated = truncate_fft_dct(x, max_freqs=max_freqs, norm='ortho')
error = torch.norm(x - x_truncated, p=2)
#print(f"Difference: {torch.abs(x-x_truncated).round(decimals=4)}")
print(f"Reconstruction error FFT (2D): {error.item()}")

# Load the x vector from the file
x = torch.load(file_path)
max_freqs = 144
x_truncated = truncate_fft_dct_sort(x, max_freqs=max_freqs, norm='ortho')
error = torch.norm(x - x_truncated, p=2)
#print(f"Difference: {torch.abs(x-x_truncated).round(decimals=4)}")
print(f"Reconstruction error FFT sort (2D): {error.item()}")

sys.exit()
# Test orthogonality with custom tolerances
rtol = 1e-4  # Relative tolerance
atol = 1e-6  # Absolute tolerance
orthogonality = torch.allclose(torch.matmul(dct_matrix, idct_matrix), torch.eye(N, device=device, dtype=dtype),rtol=rtol,
    atol=atol)
print(f"Orthogonality check: {orthogonality}")

print(f"Poduct: {torch.matmul(dct_matrix, idct_matrix).round(decimals=4)}")
print(f"Poduct2: {torch.matmul(idct_matrix, dct_matrix).round(decimals=4)}")
print(f"Inversion norm: {torch.norm(torch.matmul(dct_matrix, idct_matrix))}")

sys.exit()
# Create a test matrix
x = torch.randn(N, N)
print("Original matrix:")
print(x)

# Apply 2D DCT
x_dct = apply_dct(x)
print("\n2D DCT result:")
print(x_dct)

# Apply 2D IDCT to reconstruct
x_reconstructed = apply_idct(x_dct)
print("\nReconstructed matrix:")
print(x_reconstructed)

# Check reconstruction error
error = torch.norm(x - x_reconstructed)
print(f"\nReconstruction error: {error.item()}")
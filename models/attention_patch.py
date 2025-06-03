import torch
import math
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Sparsification functions
def sparsify_q(x, percentile=2):
    """Sparsify query matrix by keeping only values above threshold"""
    threshold = torch.quantile(x.abs(), percentile / 100.0)
    mask = (x.abs() >= threshold).float()
    return x * mask

def sparsify_k(x, percentile=2):
    """Sparsify key matrix by keeping only values above threshold"""
    threshold = torch.quantile(x.abs(), percentile / 100.0)
    mask = (x.abs() >= threshold).float()
    return x * mask

def sparsify_v(x, percentile=2):
    """Sparsify value matrix by keeping only values above threshold"""
    threshold = torch.quantile(x.abs(), percentile / 100.0)
    mask = (x.abs() >= threshold).float()
    return x * mask

# Sparse matmul utility
def sparse_dense_matmul(sparse, dense):
    """
    sparse: (N, D) sparse tensor
    dense: (D, M) dense tensor
    returns: (N, M)
    """
    return torch.sparse.mm(sparse, dense)

# Patched attention implementation
class PatchedCausalSelfAttention(torch.nn.Module):
    def __init__(self, original_attn, sparsify_q_fn, sparsify_k_fn, sparsify_v_fn, backproject_fn=None, compressed_dim=None, layers_to_patch=None):
        super().__init__()
        self.original_attn = original_attn
        self.sparsify_q_fn = sparsify_q_fn
        self.sparsify_k_fn = sparsify_k_fn
        self.sparsify_v_fn = sparsify_v_fn
        self.backproject_fn = backproject_fn

        # Get original dimensions
        self.embed_dim = getattr(original_attn, 'embed_dim', getattr(original_attn, 'split_size', 768))
        self.n_head = getattr(original_attn, 'n_head', getattr(original_attn, 'num_heads', 12))
        self.head_dim = self.embed_dim // self.n_head

        # For backward compatibility
        self.num_heads = self.n_head
        self.split_size = self.embed_dim

        # Original dimensions for reference
        self.orig_embed_dim = self.embed_dim
        self.orig_head_dim = self.head_dim

        # Track if we're using compression
        self.using_compression = False

        # Update dimensions if using compression
        if compressed_dim is not None and compressed_dim != self.embed_dim:
            self.using_compression = True
            self.embed_dim = compressed_dim
            assert compressed_dim % self.n_head == 0, f"Compressed dimension {compressed_dim} must be divisible by {self.n_head} heads"
            self.head_dim = compressed_dim // self.n_head
            self.split_size = compressed_dim

            if self.backproject_fn is None and self.using_compression:
                print("WARNING: Using compression without a backproject function!")

        # Copy other attributes from original attention
        for name, attr in vars(original_attn).items():
            if not name.startswith('_') and name not in ['c_attn', 'c_proj', 'attn_dropout', 'resid_dropout']:
                setattr(self, name, attr)

    def _split_heads(self, tensor, num_heads, head_dim=None):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        With support for variable dimensions
        """
        # Calculate head_dim if not provided
        if head_dim is None:
            head_dim = tensor.size(-1) // num_heads

        new_shape = tensor.size()[:-1] + (num_heads, head_dim)

        # Make sure the reshape is valid
        total_elements = tensor.numel()
        new_elements = np.prod([dim for dim in new_shape])

        if total_elements != new_elements:
            # Recalculate dimensions to make it work
            actual_dim = tensor.size(-1)
            actual_head_dim = actual_dim // num_heads
            new_shape = tensor.size()[:-1] + (num_heads, actual_head_dim)

        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, head_dim=None):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        With support for variable dimensions
        """
        # Get actual dimensions from tensor
        batch_size, heads, seq_len, actual_head_dim = tensor.shape

        # Calculate head_dim if not provided or different from actual
        if head_dim is None or head_dim != actual_head_dim:
            head_dim = actual_head_dim

        # Transpose back to [batch, seq, heads, head_dim]
        tensor = tensor.permute(0, 2, 1, 3).contiguous()

        # Calculate new combined dimension
        combined_dim = heads * head_dim

        # Create new shape
        new_shape = tensor.size()[:-2] + (combined_dim,)

        # Reshape
        return tensor.view(*new_shape)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        # Use original c_attn to project QKV
        qkv = self.original_attn.c_attn(hidden_states)

        # Split QKV - using 3-way split for GPT-2 style
        qkv_split_size = qkv.shape[-1] // 3
        query, key, value = qkv.split(qkv_split_size, dim=2)

        # Get batch size and sequence length
        batch_size, seq_length = hidden_states.shape[:2]

        # Handle key-value caching
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)

        # Store for caching
        present = (key, value) if use_cache else None

        # Apply compression if using it
        if self.using_compression:
            # Reshape for compression
            query_flat = query.reshape(-1, query.shape[-1])
            key_flat = key.reshape(-1, key.shape[-1])
            value_flat = value.reshape(-1, value.shape[-1])

            # Apply compression
            if callable(self.sparsify_q_fn):
                query_flat = self.sparsify_q_fn(query_flat)
            if callable(self.sparsify_k_fn):
                key_flat = self.sparsify_k_fn(key_flat)
            if callable(self.sparsify_v_fn):
                value_flat = self.sparsify_v_fn(value_flat)

            # Reshape back
            query = query_flat.reshape(batch_size, seq_length, -1)
            key = key_flat.reshape(batch_size, key.shape[1], -1)  # Key length might differ due to cache
            value = value_flat.reshape(batch_size, value.shape[1], -1)  # Value length might differ due to cache

        # Get actual dimensions after compression
        actual_q_dim = query.shape[-1]
        actual_k_dim = key.shape[-1]
        actual_v_dim = value.shape[-1]

        # Ensure dimensions are all the same (as required by attention)
        if not (actual_q_dim == actual_k_dim == actual_v_dim):
            # This shouldn't happen but handle just in case
            print(f"WARNING: Inconsistent dimensions after compression: Q={actual_q_dim}, K={actual_k_dim}, V={actual_v_dim}")

        # Calculate actual head dimensions based on compressed state
        actual_dim = actual_q_dim  # Use query dimension as reference
        actual_head_dim = actual_dim // self.n_head

        # Split heads with actual dimensions
        query = self._split_heads(query, self.n_head, actual_head_dim)
        key = self._split_heads(key, self.n_head, actual_head_dim)
        value = self._split_heads(value, self.n_head, actual_head_dim)

        # Compute attention
        attn_output, attn_weights = self._attn(
            query, key, value, attention_mask, head_mask
        )

        # Merge heads back using actual dimensions
        attn_output = self._merge_heads(attn_output, self.n_head, actual_head_dim)

        # Back-project if needed
        if self.using_compression and callable(self.backproject_fn):
            # Flatten for back-projection
            attn_output_flat = attn_output.reshape(-1, attn_output.shape[-1])
            attn_output_flat = self.backproject_fn(attn_output_flat)
            attn_output = attn_output_flat.reshape(batch_size, seq_length, -1)

        # Final projection
        attn_output = self.original_attn.c_proj(attn_output)

        # Return with appropriate outputs
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Get dimensions - note GPT-2's peculiar handling of dimensions
        batch_size, num_heads, q_len, head_dim = query.size()
        k_len = key.size(-2)

        # Compute attention scores - use exactly the same computation as GPT-2
        # [batch, heads, q_len, head_dim] x [batch, heads, head_dim, k_len]
        # -> [batch, heads, q_len, k_len]
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        # Scale exactly as in GPT-2
        attn_weights = attn_weights / math.sqrt(head_dim)

        # Apply causal mask - use exactly same implementation as GPT-2
        # Critical: Use -10000.0 instead of -inf for better numerical stability
        if q_len > 1:  # No need for causal mask when generating single token
            causal_mask = torch.tril(
                torch.ones((q_len, k_len), dtype=attn_weights.dtype, device=attn_weights.device)
            )
            mask = causal_mask.view(1, 1, q_len, k_len)
            attn_weights = attn_weights * mask + -10000.0 * (1.0 - mask)

        # Apply attention mask if provided (for padded tokens)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Standard softmax
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        # Ensure same dtype as inputs
        attn_weights = attn_weights.type_as(value)

        # Apply head mask (e.g., for pruning heads)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # Calculate context
        context = torch.matmul(attn_weights, value)

        return context, attn_weights


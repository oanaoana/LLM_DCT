import torch
import math
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

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
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
        # Ensure precision matches exactly between implementations
        dtype = hidden_states.dtype

        # During generation, we'll get a single token at a time when layer_past is provided
        is_generation_mode = layer_past is not None and hidden_states.shape[1] == 1

        # Use original c_attn to project QKV
        qkv = self.original_attn.c_attn(hidden_states)

        # Split QKV - here's where GPT-2 splits into Q, K, V
        qkv_size = qkv.size(-1)
        query, key, value = qkv.split(qkv_size // 3, dim=2)

        # Get batch size and sequence length
        batch_size, seq_length = hidden_states.shape[:2]

        # Apply sparsification functions if they're callable
        # IMPORTANT: Only apply sparsification if not in generation mode OR
        # if both current and past key/values need to be sparsified consistently
        if not is_generation_mode:
            # Regular training/inference pass - apply sparsification to full sequence
            query_flat = query.reshape(-1, query.size(-1))
            key_flat = key.reshape(-1, key.size(-1))
            value_flat = value.reshape(-1, value.size(-1))

            query_flat = self.sparsify_q_fn(query_flat) if callable(self.sparsify_q_fn) else query_flat
            key_flat = self.sparsify_k_fn(key_flat) if callable(self.sparsify_k_fn) else key_flat
            value_flat = self.sparsify_v_fn(value_flat) if callable(self.sparsify_v_fn) else value_flat

            query = query_flat.reshape(batch_size, seq_length, -1)
            key = key_flat.reshape(batch_size, seq_length, -1)
            value = value_flat.reshape(batch_size, seq_length, -1)
        else:
            # Generation mode with cached key/values
            # In this case, we need to handle sparsification carefully
            if callable(self.sparsify_q_fn):
                query_flat = query.reshape(-1, query.size(-1))
                query_flat = self.sparsify_q_fn(query_flat)
                query = query_flat.reshape(batch_size, seq_length, -1)

            if layer_past is not None:
                # If using cached key/values, apply sparsification only to the new token
                # and ensure past key/values were already sparsified consistently
                past_key, past_value = layer_past

                # Only sparsify new key/value tokens before concatenating
                if callable(self.sparsify_k_fn):
                    key_flat = key.reshape(-1, key.size(-1))
                    key_flat = self.sparsify_k_fn(key_flat)
                    key = key_flat.reshape(batch_size, seq_length, -1)

                if callable(self.sparsify_v_fn):
                    value_flat = value.reshape(-1, value.size(-1))
                    value_flat = self.sparsify_v_fn(value_flat)
                    value = value_flat.reshape(batch_size, seq_length, -1)

                # Concatenate with past key/values
                key = torch.cat((past_key, key), dim=1)
                value = torch.cat((past_value, value), dim=1)

        # Store current key/value for future calls
        present = (key, value) if use_cache else None

        # Split heads - match original implementation exactly
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # Calculate attention
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # Merge heads back
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)

        # Project back to original dimension if needed
        if self.embed_dim != self.orig_embed_dim and callable(self.backproject_fn):
            attn_output_flat = attn_output.reshape(-1, attn_output.size(-1))
            attn_output_flat = self.backproject_fn(attn_output_flat)
            attn_output = attn_output_flat.reshape(batch_size, seq_length, -1)

        # Final projection
        attn_output = self.original_attn.c_proj(attn_output)

        # Ensure output has same dtype as input for numerical stability
        attn_output = attn_output.to(dtype)

        # Return outputs
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


# Main function to apply sparsification to a model
def apply_attention_sparsification(model, sparsify_q_fn=sparsify_q, sparsify_k_fn=sparsify_k, sparsify_v_fn=sparsify_v, backproject_fn=None, compressed_dim=None, layers_to_patch=None):
    """
    Apply sparsification to the attention mechanism of a GPT-2 model.

    Args:
        model: The GPT-2 model to modify
        sparsify_q_fn: Function to sparsify query matrices
        sparsify_k_fn: Function to sparsify key matrices
        sparsify_v_fn: Function to sparsify value matrices
        backproject_fn: Function to project attention outputs back to original dimension
        compressed_dim: If specified, use compressed attention with this dimension
        layers_to_patch: List of layer indices to patch (if None, patch all)

    Returns:
        The modified model with sparsified attention
    """
    # Check and set required attributes
    for idx, block in enumerate(model.transformer.h):
        if not hasattr(block.attn, 'split_size'):
            block.attn.split_size = block.attn.embed_dim
        if not hasattr(block.attn, 'num_heads'):
            block.attn.num_heads = block.attn.n_head
        if not hasattr(block.attn, 'head_dim'):
            block.attn.head_dim = block.attn.embed_dim // block.attn.n_head
        if (layers_to_patch is None) or (idx in layers_to_patch):
            print(f"Patching layer {idx}...")
            # Patch with sparsified attention
            block.attn = PatchedCausalSelfAttention(
                block.attn,
                sparsify_q_fn=sparsify_q_fn,
                sparsify_k_fn=sparsify_k_fn,
                sparsify_v_fn=sparsify_v_fn,
                backproject_fn=backproject_fn,
                compressed_dim=compressed_dim
            )
        else:
            print(f"Skipping layer {idx}...")
    return model


# Utility functions for model evaluation
def generate_text(model, tokenizer, prompt, max_new_tokens=50, device="cuda"):
    """Generate text using the given model and prompt"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def benchmark_inference(model, tokenizer, prompt, max_new_tokens=50, device="cuda"):
    """Benchmark inference time and memory usage"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        import time
        start_mem = torch.cuda.memory_allocated() if device == "cuda" else None
        start_time = time.time()

        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

        end_time = time.time()
        end_mem = torch.cuda.memory_allocated() if device == "cuda" else None

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        results = {
            "text": generated_text,
            "time": end_time - start_time,
            "memory": (end_mem - start_mem) / 1024**2 if start_mem is not None else None
        }

        return results

def verify_attention_implementation(original_model, sparsified_model, tokenizer, device):
    """
    Verify that patched attention implementation produces the same results as original
    when using identity functions (no sparsification)
    """
    print("\n=== VERIFYING ATTENTION IMPLEMENTATION ===")

    # Set seeds for deterministic results
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    # Create identical input
    test_input = "This is a test input for verification."
    inputs = tokenizer(test_input, return_tensors="pt").to(device)

    # Get outputs from both models
    with torch.no_grad():
        # Original model
        original_outputs = original_model(**inputs)
        original_logits = original_outputs.logits

        # Patched model
        sparsified_outputs = sparsified_model(**inputs)
        sparsified_logits = sparsified_outputs.logits

    # Compare logits (output predictions)
    diff = (original_logits - sparsified_logits).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Maximum absolute difference: {max_diff:.8f}")
    print(f"Mean absolute difference: {mean_diff:.8f}")

    # Check if they're close enough
    is_equal = max_diff < 1e-4  # Using a slightly larger threshold for numerical stability
    print(f"Implementations match: {'YES' if is_equal else 'NO'}")

    # For non-matching implementations, show where differences occur
    if not is_equal:
        print("\nDifferences in attention implementation detected!")
        print("This means your patched attention may not be mathematically equivalent to the original.")
        print("Check your patched implementation for:")
        print("1. Missing or incorrect _attn method call")
        print("2. Different scaling in attention computation")
        print("3. Handling of attention mask and head mask")

    return is_equal
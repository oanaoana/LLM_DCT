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
    def __init__(self, original_attn, sparsify_q_fn, sparsify_k_fn, sparsify_v_fn, backproject_fn=None, compressed_dim=None):
        super().__init__()
        self.original_attn = original_attn
        self.sparsify_q_fn = sparsify_q_fn
        self.sparsify_k_fn = sparsify_k_fn
        self.sparsify_v_fn = sparsify_v_fn
        self.backproject_fn = backproject_fn

        # Original embedding dimensions
        self.orig_embed_dim = getattr(original_attn, 'embed_dim', original_attn.split_size)

        # Handle compression if specified
        if compressed_dim is not None:
            self.embed_dim = compressed_dim
            self.num_heads = original_attn.num_heads if hasattr(original_attn, 'num_heads') else original_attn.n_head
            assert compressed_dim % self.num_heads == 0, "Compressed dimension must be divisible by number of heads"
            self.head_dim = compressed_dim // self.num_heads
            self.split_size = compressed_dim
            self.scale_attn = 1.0 / math.sqrt(self.head_dim)

            # Check if we have a backprojection function
            if self.backproject_fn is None:
                print("WARNING: No backproject_fn provided with compressed_dim. Results may be incorrect.")
        else:
            # Fallback: no compression
            self.embed_dim = self.orig_embed_dim
            self.num_heads = getattr(original_attn, 'num_heads', original_attn.n_head)
            self.head_dim = getattr(original_attn, 'head_dim', self.embed_dim // self.num_heads)
            self.split_size = getattr(original_attn, 'split_size', self.embed_dim)
            self.scale_attn = getattr(original_attn, 'scale_attn', 1.0 / math.sqrt(self.head_dim))

        # Other parameters (caching, etc.)
        self.is_causal = getattr(original_attn, 'is_causal', True)

        # Copy all attributes from the original attention
        for attr_name in dir(original_attn):
            if not attr_name.startswith('_') and not hasattr(self, attr_name):
                setattr(self, attr_name, getattr(original_attn, attr_name))

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
        # Get the original QKV projections
        qkv = self.original_attn.c_attn(hidden_states)

        # Fix the split operation - GPT-2 concatenates q,k,v along dim 2
        # The split size should be embedding_dim (not split_size which could be compressed_dim)
        qkv_split = qkv.split(self.original_attn.embed_dim, dim=2)

        # If we got exactly 3 tensors, unpack them
        if len(qkv_split) == 3:
            query, key, value = qkv_split
        else:
            # Otherwise, the original model might have a different pattern
            # Let's try the most common GPT-2 pattern
            qkv_size = qkv.size(-1)
            query, key, value = qkv.split(qkv_size // 3, dim=2)

        # Type safety check for sparsification functions
        if not callable(self.sparsify_q_fn):
            print(f"WARNING: sparsify_q_fn is not callable, it's {type(self.sparsify_q_fn)}. Using identity function.")
            query = query  # No change
        else:
            query = self.sparsify_q_fn(query)

        if not callable(self.sparsify_k_fn):
            print(f"WARNING: sparsify_k_fn is not callable, it's {type(self.sparsify_k_fn)}. Using identity function.")
            key = key  # No change
        else:
            key = self.sparsify_k_fn(key)

        if not callable(self.sparsify_v_fn):
            print(f"WARNING: sparsify_v_fn is not callable, it's {type(self.sparsify_v_fn)}. Using identity function.")
            value = value  # No change
        else:
            value = self.sparsify_v_fn(value)

        # Handle past key/values for generation
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)

        present = (key, value) if use_cache else None

        # Split heads
        batch_size, seq_length = query.shape[:2]
        head_dim = self.head_dim
        n_head = self.num_heads

        # Reshape for multi-head attention
        query = query.view(batch_size, seq_length, n_head, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, n_head, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, n_head, head_dim).transpose(1, 2)

        # Calculate attention
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)

        # Project output back to original dimensions if needed
        if self.embed_dim != self.orig_embed_dim:
            if callable(self.backproject_fn):
                attn_output = self.backproject_fn(attn_output)  # Project back to original dimension
            else:
                print("WARNING: Missing backproject_fn for compressed attention. Using identity.")
                # We should never reach here in production; compressed attention requires backprojection

        # Project output using original projection
        attn_output = self.original_attn.c_proj(attn_output)

        # Return based on flags
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    def _attn(self, q, k, v, attention_mask=None, head_mask=None):
        # Get sequence length and batch size
        bsz, num_heads, q_len, head_dim = q.size()
        k_len = k.size(2)  # Could be different from q_len due to past_key

        # Calculate attention scores
        # (bsz, num_heads, q_len, k_len)
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(head_dim)  # Scale using head_dim

        # Apply causal mask if needed (for autoregressive generation)
        if q_len > 1:
            # PyTorch masking is done with -inf
            causal_mask = torch.triu(
                torch.ones((q_len, k_len), dtype=torch.bool, device=q.device),
                diagonal=1
            )
            attn_weights.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Apply attention mask if provided
        if attention_mask is not None:
            # Add attention mask to scores
            attn_weights = attn_weights + attention_mask

        # Apply softmax
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        # Apply head mask if provided
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # Calculate context
        attn_output = torch.matmul(attn_weights, v)

        return attn_output, attn_weights


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

import torch
from models.attention_patch import PatchedCausalSelfAttention

# Main function to apply sparsification to a model
def apply_attention_sparsification(model, sparsify_q_fn=None, sparsify_k_fn=None, sparsify_v_fn=None, backproject_fn=None, compressed_dim=None, layers_to_patch=None):
    """
    Apply sparsification to the attention mechanism of a GPT-2 model.
    """
    # Print configuration for debugging
    print(f"Applying attention sparsification:")
    print(f" - Compressed dimension: {compressed_dim}")
    print(f" - Layers to patch: {layers_to_patch}")
    print(f" - Using backprojection: {backproject_fn is not None}")

    # Check and set required attributes
    for idx, block in enumerate(model.transformer.h):
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
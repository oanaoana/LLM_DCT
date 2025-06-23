# LLM_DCT - Attention Sparsification for GPT-2 Models

This project studies attention sparsification strategies for GPT-2 models using transform-based compression techniques. The focus is on reducing computational complexity while maintaining model performance through efficient sparse matrix operations.

## Overview

We investigate various transform-based approaches to sparsify attention mechanisms in large language models, specifically targeting the Key (K) and Value (V) matrices in GPT-2's multi-head attention layers. The project explores both Discrete Cosine Transform (DCT) and Prolate Spheroidal Wave Function (PSWF) bases for frequency-domain compression.

## Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd LLM_DCT

# Install dependencies
pip install -r requirements.txt

# Run basic DCT compression
python main_final.py --transform dct --compression 0.8

# Run full evaluation
python main_final.py --evaluate --benchmark --verify
```

## Core Features

- **DCT-based attention compression** with configurable compression ratios
- **PSWF transform support** for alternative basis functions
- **Layer-selective patching** to apply compression to specific transformer layers
- **Comprehensive evaluation suite** including quality metrics and performance benchmarks
- **GPU/CPU compatibility** with automatic device detection

## Usage

### Basic Commands

```bash
# Basic DCT compression with 80% compression ratio
python main_final.py --transform dct --compression 0.8

# PSWF compression on specific layers
python main_final.py --transform pswf --compression 0.7 --layers 0,1,2

# Full evaluation with benchmarking
python main_final.py --evaluate --benchmark --verify

# Force CPU mode (useful for debugging)
python main_final.py --cpu --evaluate
```

### Command Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model` | str | 'gpt2' | Model name |
| `--transform` | str | 'dct' | Transform type ('dct' or 'pswf') |
| `--compression` | float | 0.8 | Compression ratio (0.0-1.0) |
| `--layers` | str | 'all' | Layers to patch (comma-separated or 'all') |
| `--verify` | flag | False | Run verification tests |
| `--benchmark` | flag | False | Run performance benchmarks |
| `--evaluate` | flag | False | Run quality evaluation |
| `--cpu` | flag | False | Force CPU mode even if GPU is available |

## Project Structure

```
LLM_DCT/
│
├── models/
│   ├── attention_patch.py       # Patched attention implementation
│   ├── sparsification.py        # Sparsification methods (PSWF, DCT, etc.)
│   └── model_utils.py           # Model loading, patching, and basic utilities
│
├── transforms/
│   ├── dct.py                   # DCT implementation
│   ├── pswf.py                  # PSWF implementation
│   └── utils.py                 # Shared transform utilities
│
├── evaluation/
│   ├── benchmark.py             # Performance benchmarking
│   ├── metrics.py               # Quality metrics (BLEU, etc.)
│   └── verification.py          # Implementation verification
│
├── utils/                       # General utilities and helpers
├── examples/                    # Example scripts and tutorials
├── tests/                       # Test suite
│   ├── conftest.py             # Pytest fixtures
│   ├── test_transforms.py      # Transform tests
│   ├── test_attention.py       # Attention patching tests
│   └── test_integration.py     # End-to-end tests
│
├── main_final.py               # Main execution pipeline
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore patterns
└── README.md                   # This file
```

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- CUDA (optional, for GPU acceleration)

### Dependencies

```bash
pip install -r requirements.txt
```

## Testing

### Run Test Suite

```bash
# Run all tests
python -m pytest tests/

# Run specific test modules
python -m pytest tests/test_transforms.py
python -m pytest tests/test_integration.py

# Run with verbose output
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=models --cov=transforms --cov=evaluation

# Run fast tests only (skip slow ones)
python -m pytest tests/ -k "not slow"

# Run specific test function
python -m pytest tests/test_integration.py::test_end_to_end_pipeline

# Run specific parametrized test case
python -m pytest tests/test_integration.py::test_layer_patching_configurations[all]
```

## Examples and Scripts

### Compression Benchmarks

```bash
# Run all methods with default compression ratio (0.5)
python scripts/compression_benchmark.py --method all

# Run just the per-vector method
python scripts/compression_benchmark.py --method per_vector

# Try different compression ratios
python scripts/compression_benchmark.py --method layer_wise --compression_ratio 0.25

# Specify output directory
python scripts/compression_benchmark.py --output_dir results/pswf_benchmark
```

### Transform Diagnostics

```bash
# To diagnose DCT only
python scripts/diagnose_transforms.py dct

# To diagnose PSWF only
python scripts/diagnose_transforms.py pswf
```

### Evaluation Examples

```bash
# Main pipeline with evaluation
python main_final.py --transform dct --compression 0.7 --benchmark --evaluate

# Standalone benchmark across multiple compression ratios
python -m evaluation.benchmark --model gpt2 --compression 0.5,0.7,0.9

# Run compression study examples
python examples/compression_study.py
```

## Research Goals & TODO

### 1. Blockwise DCT with Boundary Corrections ⏳
- [ ] Implement different boundary condition strategies for block-wise DCT
- [ ] Study overlap vs. non-overlap block processing
- [ ] Analyze reconstruction quality across different boundary treatments
- [ ] Optimize block size selection for various model architectures

### 2. PSWF Stability Study 🔬
- [ ] Investigate numerical stability of PSWF basis construction
- [ ] Compare PSWF vs. DCT compression effectiveness
- [ ] Analyze frequency localization properties for attention matrices
- [ ] Develop adaptive basis selection strategies

### 3. Sparse Matrix-Vector (SpMV) Kernel Development ⚡
- [ ] Implement custom SpMV kernels for compressed attention operations
- [ ] Optimize memory access patterns for sparse K/V matrices
- [ ] Develop GPU kernels for batched sparse operations
- [ ] Benchmark against existing sparse libraries (cuSPARSE, MKL)

### 4. LLVM Implementation & Comparison 🏗️
- [ ] Develop LLVM-based sparse matrix multiplication kernels
- [ ] Create CPU-optimized implementations for comparison
- [ ] Benchmark against cuSPARSE performance on GPU
- [ ] Study compilation optimizations for specific sparsity patterns
- [ ] Analyze trade-offs between compilation time and runtime performance

### Development Guidelines

- Add tests for any new functionality
- Follow existing code style and formatting
- Update documentation as needed
- Ensure all tests pass before submitting PR

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{llm_dct_2024,
  title={Attention Sparsification Strategies for GPT-2 Models},
  author={Oana Marin},
  year={2025},
  howpublished={\url{https://github.com/oanaoana/LLM_DCT}}
}
```

## Contact

- **Author**: Oana Marin
- **Project Link**: [https://github.com/oanaoana/LLM_DCT](https://github.com/oanaoana/LLM_DCT)

---

**Status**: 🚧 Active Development | **Last Updated**: May 2025
# Pseudocode to C++ Transformer

A bidirectional neural machine translation system that converts between pseudocode and C++ using a custom-built Transformer architecture with multi-head attention mechanisms.

## Overview

This project implements a sequence-to-sequence model based on the Transformer architecture to translate natural language pseudocode into executable C++ code. The model leverages self-attention and cross-attention mechanisms to understand the semantic structure of pseudocode and generate syntactically correct C++ implementations.

## Features

- **Bidirectional Translation**: Convert pseudocode to C++ and vice versa
- **Custom Transformer Architecture**: Built from scratch with multi-head attention
- **Interactive Web Interface**: Easy-to-use Gradio-based chat interface
- **Streaming Output**: Real-time code generation with token-by-token display
- **SentencePiece Tokenization**: Efficient subword tokenization for both languages
- **GPU Acceleration**: Automatic CUDA support for faster inference

## Architecture

The model consists of:

- **Multi-Head Attention Layers**: Parallel attention mechanisms for capturing different aspects of input sequences
- **Positional Encoding**: Sinusoidal encoding to inject sequence order information
- **Encoder-Decoder Structure**:
  - Encoder processes input pseudocode with self-attention
  - Decoder generates C++ code with masked self-attention and cross-attention
- **Position-wise Feed-Forward Networks**: Deep feature transformation after attention layers
- **Layer Normalization & Dropout**: Regularization for stable training

### Model Parameters

- **Embedding Dimension**: 512 (d_model)
- **Attention Heads**: 8
- **Feed-Forward Dimension**: 2048
- **Encoder/Decoder Layers**: 6 each
- **Maximum Sequence Length**: Configurable
- **Dropout Rate**: 0.1

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0+
- CUDA-capable GPU (optional, but recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure model files are present:
- `pseudo-to-cpp-model.pth` - Trained transformer weights
- `pseudo.model` / `pseudo.vocab` - SentencePiece model for pseudocode
- `code.model` / `code.vocab` - SentencePiece model for C++

## Usage

### Running the Web Interface

Launch the Gradio interface:

```bash
python app.py
```

The interface will be available at `http://localhost:7860`

### Example Inputs

**Pseudocode:**
```
for i from 1 to n, print i
```

**Generated C++:**
```cpp
for(int i = 1; i <= n; i++) {
    cout << i << endl;
}
```

### Programmatic Usage

```python
import torch
import sentencepiece as spm

# Load model and tokenizers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("pseudo-to-cpp-model.pth", map_location=device)
sp_pseudo = spm.SentencePieceProcessor(model_file="pseudo.model")
sp_code = spm.SentencePieceProcessor(model_file="code.model")

# Generate code
pseudocode = "initialize variable x to 0"
src = torch.tensor([sp_pseudo.encode_as_ids(pseudocode)], device=device)
tgt = torch.tensor([[2]], device=device)  # BOS token

with torch.no_grad():
    output = model(src, tgt)
    # Process output tokens...
```

## Model Training

The model was trained on the SPoC (Pseudocode-to-Code) dataset, which contains paired examples of pseudocode descriptions and their corresponding C++ implementations.

### Training Features

- **Tokenization**: SentencePiece with BPE algorithm
- **Optimization**: Adam optimizer with learning rate scheduling
- **Loss Function**: Cross-entropy loss with label smoothing
- **Masking**: Causal masking for autoregressive decoding
- **Regularization**: Dropout and layer normalization

## Technical Details

### Attention Mechanism

The scaled dot-product attention computes:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Where Q, K, V are query, key, and value matrices derived from input embeddings.

### Positional Encoding

Position information is injected using sine and cosine functions:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Generation Strategy

- **Autoregressive Decoding**: Generate one token at a time
- **Greedy Search**: Select highest probability token at each step
- **End-of-Sequence Detection**: Stop when EOS token (ID: 5) is generated
- **Maximum Length Constraint**: Prevent infinite generation

## Project Structure

```
.
├── app.py                      # Gradio web application
├── pseudo-to-cpp-model.pth     # Trained model weights
├── pseudo.model                # Pseudocode tokenizer
├── pseudo.vocab                # Pseudocode vocabulary
├── code.model                  # C++ tokenizer
├── code.vocab                  # C++ vocabulary
├── data.txt                    # Training dataset
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Dependencies

- **PyTorch**: Deep learning framework for model implementation
- **SentencePiece**: Unsupervised tokenization library
- **Gradio**: Web interface for interactive demonstrations
- **NumPy**: Numerical computing library
- **Pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning utilities
- **tqdm**: Progress bars for training loops

## Performance Considerations

- **Inference Speed**: ~0.5-2s per translation (depends on sequence length and hardware)
- **Memory Requirements**: ~2GB GPU memory for inference
- **Batch Processing**: Model supports batched inference for multiple translations
- **Optimization**: Model uses CUDA when available for 10-50x speedup

## Limitations

- Quality depends on training data diversity
- Long sequences may experience degraded performance
- May generate syntactically valid but semantically incorrect code
- Limited to C++ language constructs present in training data

## Future Improvements

- [ ] Add beam search decoding for better output quality
- [ ] Support additional programming languages (Python, Java, JavaScript)
- [ ] Implement attention visualization
- [ ] Fine-tune on domain-specific codebases
- [ ] Add code syntax validation and error correction
- [ ] Support reverse translation (C++ to pseudocode)

## Tech Stack

**Machine Learning**: PyTorch, Transformer, Seq2Seq, Multi-Head Attention

**Tokenization**: SentencePiece (BPE)

**Interface**: Gradio

**Language**: Python 3.8+

## License

This project is available for educational and research purposes.

## Acknowledgments

- Based on "Attention Is All You Need" (Vaswani et al., 2017)
- Trained on the SPoC dataset
- Inspired by modern neural machine translation systems

## Citation

If you use this work, please cite:

```bibtex
@misc{pseudocode-to-cpp-transformer,
  title={Pseudocode to C++ Transformer},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/pseudocode-to-cpp}}
}
```

---

**Note**: This model is a research prototype. Generated code should be reviewed and tested before production use.

# MiniGPT — Training a GPT Language Model from Scratch

A hands-on implementation of language models using **PyTorch**, inspired by [Andrej Karpathy's Neural Networks: Zero to Hero video lectures](https://github.com/karpathy/ng-video-lecture).

This project explores how language models learn to predict the next token by implementing two architectures from scratch:

1. **Bigram Language Model** — A simple character-level language model.
2. **GPT Transformer** — A decoder-only Transformer architecture trained using the OpenWebText dataset and GPT-2 BPE tokenization.

The objective is to understand the fundamentals of language modeling, tokenization, self-attention, Transformer architecture, and autoregressive text generation.

---

## Project Overview

The project follows a progressive approach to building a language model.

| Model | Description |
|---|---|
| Bigram Language Model | Predicts the next character based on the current character |
| GPT Transformer | Predicts the next token using contextual information and self-attention |
| GPT-2 Tokenizer | Converts text into subword token IDs using Byte Pair Encoding |
| OpenWebText | Provides a large-scale English web-text training corpus |

## Architecture

### 1. Bigram Language Model

The Bigram model is the simplest language model implemented in this project.

It learns the probability of the next character based on the current character.

**Architecture:**

- Character-level tokenization
- Character-to-integer encoding
- Embedding lookup table
- Cross-entropy loss
- AdamW optimizer
- Autoregressive text generation

The model learns patterns in consecutive characters but cannot effectively capture long-range dependencies.

### 2. GPT Transformer

The GPT implementation is based on the decoder-only Transformer architecture introduced in Andrej Karpathy's educational implementation.

It consists of:

- Token Embeddings
- Positional Embeddings
- Masked Multi-Head Self-Attention
- Feed-Forward Neural Networks
- Residual Connections
- Layer Normalization
- Language Modeling Head

**Transformer architecture:**

```text
Input Tokens
     |
     v
Token Embeddings
     |
     +
Positional Embeddings
     |
     v
Transformer Blocks
     |
     +-- Masked Multi-Head Attention
     |
     +-- Feed Forward Network
     |
     v
Layer Normalization
     |
     v
Linear Language Modeling Head
     |
     v
Next Token Prediction
```

The model uses causal attention, ensuring that each token can only attend to previous tokens and itself.

---

## Dataset — OpenWebText

This project uses [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext), an open dataset of web documents inspired by the WebText dataset used in GPT-2 research.

Dataset source:

https://huggingface.co/datasets/Skylion007/openwebtext

The dataset is downloaded using the Hugging Face `datasets` library with streaming enabled.

The current data preparation script extracts up to 100,000 documents and combines their text into `input.txt`.

### Data Preparation Pipeline

```text
OpenWebText Dataset
        |
        v
Hugging Face Datasets
        |
        v
Streaming Data Extraction
        |
        v
Document Text Extraction
        |
        v
input.txt
        |
        v
Tokenization
        |
        v
Training / Validation Split
```

### Download the Dataset

Run:

```bash
python download_data.py
```

This generates:

```text
input.txt
```

Note: The current script uses a 100,000-document limit. This is an experimental subset, not the complete OpenWebText corpus.

---

## Tokenization — GPT-2 BPE using tiktoken

The GPT Transformer uses the GPT-2 tokenizer provided by OpenAI's `tiktoken` library.

Unlike character-level tokenization, Byte Pair Encoding (BPE) represents text using reusable subword tokens.

**Tokenizer configuration:**

- Encoding: `gpt2`
- Vocabulary size: 50,257
- Tokenization: Byte Pair Encoding
- Output: Integer token IDs

Example:

```python
import tiktoken

enc = tiktoken.get_encoding("gpt2")

text = "Artificial Intelligence is amazing!"

tokens = enc.encode_ordinary(text)

print("Token IDs:", tokens)
print("Decoded:", enc.decode(tokens))
```

The tokenizer converts text into token IDs that are used as inputs to the GPT Transformer.

The Bigram model retains character-level tokenization as a separate baseline experiment.

---

## Project Structure

```text
MiniGPT/
│
├── bigram.py
│
├── gpt.py
│
├── download_data.py
│
├── input.txt
│
├── checkpoints/
│   ├── best_gpt.pt
│   └── final_gpt.pt
│
├── generated.txt
│
└── README.md
```

| File | Description |
|---|---|
| `bigram.py` | Character-level Bigram Language Model |
| `gpt.py` | GPT Transformer implementation with GPT-2 tokenization |
| `download_data.py` | Downloads and extracts OpenWebText documents |
| `input.txt` | Extracted training corpus |
| `checkpoints/` | Saved model checkpoints |
| `generated.txt` | Generated text output |

---

## Requirements

- Python 3.10+
- PyTorch
- NVIDIA GPU (recommended)
- CUDA-compatible PyTorch installation
- Hugging Face Datasets
- tiktoken
- tqdm

### Installation

Install the dependencies:

```bash
pip install torch numpy tiktoken datasets tqdm
```

For NVIDIA GPU training, install the PyTorch build compatible with your CUDA environment.

Verify CUDA availability:

```python
import torch

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

---

## Training

### 1. Train the Bigram Model

The Bigram model uses character-level tokenization.

Run:

```bash
python bigram.py
```

The training loop calculates training and validation loss periodically.

Example output:

```text
step 0: train loss 9.2086, val loss 9.2155
step 300: train loss 4.9295, val loss 4.9147
step 600: train loss 3.1509, val loss 3.2030
step 900: train loss 2.8406, val loss 2.8378
```

### 2. Prepare OpenWebText

Run:

```bash
python download_data.py
```

This extracts the selected OpenWebText documents into `input.txt`.

### 3. Train the GPT Transformer

Run:

```bash
python gpt.py
```

The GPT implementation:

1. Loads the training corpus.
2. Tokenizes the text using GPT-2 BPE.
3. Splits the dataset into training and validation sets.
4. Initializes the Transformer model.
5. Trains using next-token prediction.
6. Evaluates training and validation loss.
7. Saves model checkpoints.
8. Generates text using a user-provided prompt.

---

## GPT Training Configuration

The initial experimental configuration is:

| Hyperparameter | Value |
|---|---:|
| Vocabulary size | 50,257 |
| Batch size | 4 |
| Context length | 128 |
| Embedding dimension | 384 |
| Attention heads | 6 |
| Transformer layers | 6 |
| Dropout | 0.2 |
| Learning rate | 0.0003 |
| Training iterations | 5,000 |
| Optimizer | AdamW |

These are initial experimental settings and can be adjusted according to GPU memory and training performance.

---

## Text Generation

The GPT model generates text autoregressively.

Example prompt:

```python
prompt = "The future of artificial intelligence"
```

The prompt is converted into GPT-2 token IDs before being passed into the model.

Generation parameters include:

- `max_new_tokens`: Maximum number of new tokens.
- `temperature`: Controls randomness during sampling.
- `top_k`: Restricts sampling to the most likely tokens.

Example:

```python
context = torch.tensor(
    [encode(prompt)],
    dtype=torch.long,
    device=device
)

generated = model.generate(
    context,
    max_new_tokens=300,
    temperature=0.8,
    top_k=40
)

print(decode(generated[0].tolist()))
```

Generated output is saved in `generated.txt`.

---

## Checkpointing

The GPT training implementation saves:

- `best_gpt.pt` — Model checkpoint with the lowest observed validation loss.
- `final_gpt.pt` — Model state after the final training iteration.

These checkpoints allow the trained model weights to be loaded later for inference or further development.

---

## Hardware

Development environment:

- Operating System: Windows
- Framework: PyTorch
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU
- Compute: CUDA
- Development Environment: Visual Studio Code

The initial model configuration is designed for experimentation on a laptop GPU. Larger datasets and longer training runs require additional GPU memory, storage, and training time.

---

## Learning Objectives

This project explores the following concepts:

1. Character-level language modeling.
2. Bigram probability modeling.
3. Tokenization and vocabulary construction.
4. Byte Pair Encoding (BPE).
5. Token and positional embeddings.
6. Self-attention mechanisms.
7. Causal masking.
8. Multi-head attention.
9. Transformer blocks.
10. Cross-entropy loss.
11. Backpropagation and gradient descent.
12. Autoregressive text generation.
13. GPU-accelerated model training.

---

## Future Improvements

- [ ] Implement efficient binary token storage (`train.bin` and `val.bin`).
- [ ] Add memory-mapped dataset loading.
- [ ] Train on a larger OpenWebText subset.
- [ ] Scale training to the complete OpenWebText corpus.
- [ ] Implement mixed-precision training.
- [ ] Add training checkpoint resumption.
- [ ] Implement a terminal-based chatbot interface.
- [ ] Add training loss visualization.
- [ ] Experiment with larger context lengths.
- [ ] Optimize GPU memory utilization.

---

## References

- [Andrej Karpathy — Neural Networks: Zero to Hero](https://github.com/karpathy/ng-video-lecture)
- [OpenWebText Dataset](https://huggingface.co/datasets/Skylion007/openwebtext)
- [OpenAI tiktoken](https://github.com/openai/tiktoken)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://openai.com/index/better-language-models/)

---

## Author

Developed as a hands-on learning project to understand and implement language models and GPT architectures from scratch using PyTorch.

**Built with Python, PyTorch, OpenWebText and tiktoken.**


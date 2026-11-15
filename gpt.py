
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

# =====================================================
# 1. HYPERPARAMETERS
# =====================================================

batch_size = 4
block_size = 128

max_iters = 5000
eval_interval = 500
eval_iters = 50

learning_rate = 3e-4

n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU")

# =====================================================
# 2. TOKENIZER - GPT-2 BPE
# =====================================================

print("\nLoading GPT-2 tokenizer...")

enc = tiktoken.get_encoding("gpt2")

vocab_size = enc.n_vocab

def encode(text):
    return enc.encode_ordinary(text)

def decode(tokens):
    return enc.decode(tokens)

print("Vocabulary size:", vocab_size)

# =====================================================
# 3. LOAD OPENWEBTEXT DATASET
# =====================================================

print("\nLoading input.txt...")

if not os.path.exists("input.txt"):
    raise FileNotFoundError(
        "input.txt not found! Run download_data.py first."
    )

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Characters:", len(text))

print("\nTokenizing dataset...")

token_ids = encode(text)

print("Total tokens:", len(token_ids))

if len(token_ids) <= block_size + 1:
    raise ValueError("Dataset is too small for the selected block_size.")

data = torch.tensor(token_ids, dtype=torch.long)

# Train / validation split
n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

print("Training tokens:", len(train_data))
print("Validation tokens:", len(val_data))

del text
del token_ids

# =====================================================
# 4. DATA LOADER
# =====================================================

def get_batch(split):

    data = train_data if split == 'train' else val_data

    ix = torch.randint(
        len(data) - block_size - 1,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + block_size] for i in ix
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1] for i in ix
    ])

    x = x.to(device)
    y = y.to(device)

    return x, y

# =====================================================
# 5. LOSS ESTIMATION
# =====================================================

@torch.no_grad()
def estimate_loss():

    out = {}

    model.eval()

    for split in ['train', 'val']:

        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):

            X, Y = get_batch(split)

            logits, loss = model(X, Y)

            losses[k] = loss.item()

        out[split] = losses.mean().item()

    model.train()

    return out

# =====================================================
# 6. SELF ATTENTION HEAD
# =====================================================

class Head(nn.Module):

    def __init__(self, head_size):

        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer(
            'tril',
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1)
        wei = wei * (k.shape[-1] ** -0.5)

        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float('-inf')
        )

        wei = F.softmax(wei, dim=-1)

        wei = self.dropout(wei)

        v = self.value(x)

        out = wei @ v

        return out

# =====================================================
# 7. MULTI HEAD ATTENTION
# =====================================================

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):

        super().__init__()

        self.heads = nn.ModuleList([
            Head(head_size) for _ in range(num_heads)
        ])

        self.proj = nn.Linear(
            head_size * num_heads,
            n_embd
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        out = torch.cat(
            [h(x) for h in self.heads],
            dim=-1
        )

        out = self.dropout(self.proj(out))

        return out

# =====================================================
# 8. FEED FORWARD NETWORK
# =====================================================

class FeedForward(nn.Module):

    def __init__(self, n_embd):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(n_embd, 4 * n_embd),

            nn.ReLU(),

            nn.Linear(4 * n_embd, n_embd),

            nn.Dropout(dropout)
        )

    def forward(self, x):

        return self.net(x)

# =====================================================
# 9. TRANSFORMER BLOCK
# =====================================================

class Block(nn.Module):

    def __init__(self, n_embd, n_head):

        super().__init__()

        head_size = n_embd // n_head

        self.sa = MultiHeadAttention(
            n_head,
            head_size
        )

        self.ffwd = FeedForward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):

        x = x + self.sa(self.ln1(x))

        x = x + self.ffwd(self.ln2(x))

        return x

# =====================================================
# 10. GPT LANGUAGE MODEL
# =====================================================

class GPTLanguageModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.token_embedding_table = nn.Embedding(
            vocab_size,
            n_embd
        )

        self.position_embedding_table = nn.Embedding(
            block_size,
            n_embd
        )

        self.blocks = nn.Sequential(
            *[
                Block(n_embd, n_head)
                for _ in range(n_layer)
            ]
        )

        self.ln_f = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(
            n_embd,
            vocab_size
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):

            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):

            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(self, idx, targets=None):

        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)

        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )

        x = tok_emb + pos_emb

        x = self.blocks(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        if targets is None:

            loss = None

        else:

            B, T, C = logits.shape

            logits = logits.reshape(B * T, C)

            targets = targets.reshape(B * T)

            loss = F.cross_entropy(
                logits,
                targets
            )

        return logits, loss

    # =================================================
    # 11. TEXT GENERATION
    # =================================================

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=0.8,
        top_k=40
    ):

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond)

            logits = logits[:, -1, :]

            logits = logits / temperature

            if top_k is not None:

                values, indices = torch.topk(
                    logits,
                    min(top_k, logits.size(-1))
                )

                logits[logits < values[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(
                probs,
                num_samples=1
            )

            idx = torch.cat(
                (idx, idx_next),
                dim=1
            )

        return idx

# =====================================================
# 12. INITIALIZE MODEL
# =====================================================

print("\nInitializing GPT model...")

model = GPTLanguageModel().to(device)

print(
    "Parameters:",
    sum(p.numel() for p in model.parameters()) / 1e6,
    "Million"
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

# =====================================================
# 13. CHECKPOINT DIRECTORY
# =====================================================

os.makedirs("checkpoints", exist_ok=True)

best_val_loss = float('inf')

# =====================================================
# 14. TRAINING LOOP
# =====================================================

print("\nStarting GPT training...\n")

for iter in range(max_iters):

    # Evaluate training and validation loss

    if iter % eval_interval == 0 or iter == max_iters - 1:

        losses = estimate_loss()

        print(
            f"step {iter}: "
            f"train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )

        # Save best checkpoint

        if losses['val'] < best_val_loss:

            best_val_loss = losses['val']

            torch.save(
                {
                    'step': iter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'vocab_size': vocab_size
                },
                "checkpoints/best_gpt.pt"
            )

            print("Best checkpoint saved!")

    # Get training batch

    xb, yb = get_batch('train')

    # Forward pass

    logits, loss = model(xb, yb)

    # Backpropagation

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0
    )

    optimizer.step()

# Save final model

torch.save(
    {
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'block_size': block_size,
        'n_embd': n_embd,
        'n_head': n_head,
        'n_layer': n_layer
    },
    "checkpoints/final_gpt.pt"
)

print("\nTraining completed!")

# =====================================================
# 15. GENERATE TEXT
# =====================================================

model.eval()

prompt = "The future of artificial intelligence"

context = torch.tensor(
    [encode(prompt)],
    dtype=torch.long,
    device=device
)

print("\nPrompt:", prompt)
print("\nGenerated text:\n")

generated = model.generate(
    context,
    max_new_tokens=300,
    temperature=0.8,
    top_k=40
)

output = decode(generated[0].tolist())

print(output)

with open(
    "generated.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(output)

print("\nGenerated text saved to generated.txt")

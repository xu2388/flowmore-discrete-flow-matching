"""Reproduce the FlowMore discrete-flow-matching name generation experiment
(identical model/training/sampling as flowmore.py) but additionally:
  - record the full training loss curve
  - save the 8 generated names and their full sampling trajectory
  - draw two report figures:
      1) training_loss_curve.png
      2) name_sampling_trajectories.png  (parallel de-noising grid per name)
Outputs go under ./outcomes/.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

# ---- identical setup to flowmore.py ----
torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
stoi['_'] = 27
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)
max_len = 16


def encode(w):
    w = w + '.' + '_' * (max_len - len(w) - 1)
    return torch.tensor([stoi[ch] for ch in w]).long()


data = torch.stack([encode(w) for w in words])
n_names = len(data)
print(f"names={n_names}, vocab_size={vocab_size}, max_len={max_len}")

OUT = os.path.dirname(os.path.abspath(__file__)) + "/outcomes"
os.makedirs(OUT, exist_ok=True)

# ---- model (identical structure) ----
class MLP(nn.Module):
    def __init__(self, dims, activation=torch.nn.SiLU()):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dims, 4 * dims), nn.SiLU(), nn.Linear(4 * dims, dims))

    def forward(self, x):
        return self.net(x)


class FourierEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0
        self.register_buffer('weights_f', torch.randn(1, dim // 2))

    def forward(self, t):
        f = 2 * torch.pi * self.weights_f * t.unsqueeze(-1)
        return torch.cat((torch.cos(f), torch.sin(f)), dim=1)


class MHA(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.scale = heads, (dim // heads) ** 0.5
        self.qkv, self.out_ff = nn.Linear(dim, dim * 3, bias=False), nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        Q, K, V = self.qkv(x).chunk(3, dim=-1)
        hQ = rearrange(Q, 'b n (h d) -> (b h) n d', h=self.heads)
        hK = rearrange(K, 'b n (h d) -> (b h) n d', h=self.heads)
        hV = rearrange(V, 'b n (h d) -> (b h) n d', h=self.heads)
        attn = torch.softmax(torch.einsum('bid,bjd->bij', hQ, hK) / self.scale, dim=-1)
        Z = torch.einsum('bij,bjd->bid', attn, hV)
        return self.out_ff(rearrange(Z, '(b h) n d -> b n (h d)', h=self.heads))


class DiffusionTransformerLayer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1, self.attn = nn.LayerNorm(dim), MHA(dim, heads)
        self.norm2, self.ff = nn.LayerNorm(dim), MLP(dim)
        self.modulation = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x, c):
        g1, b1, a1, g2, b2, a2 = self.modulation(c).unsqueeze(1).chunk(6, dim=-1)
        x = x + self.attn(self.norm1(x) * (1 + g1) + b1) * a1
        x = x + self.ff(self.norm2(x) * (1 + g2) + b2) * a2
        return x


class DiffusionTransformer(nn.Module):
    def __init__(self, num_layers, dim, heads, n_tokens, vocab_size):
        super().__init__()
        self.token_embedder = nn.Embedding(vocab_size, dim)
        self.pos_encoding = nn.Parameter(torch.randn(n_tokens, dim))
        self.time_embedder = FourierEncoder(dim)
        self.layer_list = nn.ModuleList([DiffusionTransformerLayer(dim, heads) for _ in range(num_layers)])
        self.to_logits = nn.Linear(dim, vocab_size)

    def forward(self, x, t):
        x = self.token_embedder(x) + self.pos_encoding.unsqueeze(0)
        c = self.time_embedder(t)
        for layer in self.layer_list:
            x = layer(x, c)
        return self.to_logits(x)


# ---- training (identical hyper-params) ----
batch_size = 256
model = DiffusionTransformer(num_layers=3, dim=64, heads=4, n_tokens=max_len, vocab_size=vocab_size).to(device)
optim = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=10000, eta_min=1e-6)
total = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {total:,}")

loss_list = []
for step in range(10000):
    idx = torch.randint(0, len(data), (batch_size,))
    x_1 = data[idx]
    x_0 = torch.randint(0, vocab_size, x_1.shape)
    t = torch.rand(batch_size)
    x_t = torch.where(torch.rand(batch_size, max_len) < t[:, None], x_1, x_0)
    logits = model(x_t.long().to(device), t.to(device))
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), x_1.flatten(0, 1).squeeze().to(device))
    optim.zero_grad()
    loss.backward()
    optim.step()
    scheduler.step()
    loss_list.append(loss.item())

# save loss curve data + figure
np.save(os.path.join(OUT, "train_loss.npy"), np.array(loss_list))
fig, ax = plt.subplots(figsize=(8, 4.4))
steps = np.arange(1, 10001)
step_d = steps[::25]
loss_d = np.array(loss_list)[::25]
ax.plot(step_d, loss_d, color="#2a5fd0")
ax.set_xlabel("training step")
ax.set_ylabel("cross-entropy loss")
ax.set_title("FlowMore training loss (Discrete Flow Matching, 10,000 steps)")
ax.set_xlim(0, 10000)
ax.grid(alpha=0.3)
plt.tight_layout()
loss_fig = os.path.join(OUT, "training_loss_curve.png")
plt.savefig(loss_fig, dpi=150)
plt.close(fig)
print("Saved", loss_fig)

# ---- sampling: record the full trajectory ----
model.eval()
num_samples = 8
trajectories = []
final_names = []
with torch.no_grad():
    for s in range(num_samples):
        x_t = torch.randint(0, vocab_size, (1, max_len)).long()
        traj = []
        t = 0.0
        while t < 1.0 - 1e-3:
            p1 = torch.softmax(model(x_t.to(device), torch.ones(1).to(device) * t), dim=-1).cpu()
            h = min(0.1, 1.0 - t)
            one_hot_x_t = nn.functional.one_hot(x_t, vocab_size).float()
            u = (p1 - one_hot_x_t) / (1.0 - t)
            x_t = torch.distributions.Categorical(probs=one_hot_x_t + h * u).sample()
            t += h
            traj.append((t, x_t.squeeze().clone()))
        trajectories.append(traj)
        final = ''.join([itos[i] for i in x_t.squeeze().tolist()])
        final_names.append(final)
        print(f"sample {s}: {final}")

with open(os.path.join(OUT, "generated_names.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(final_names))

# ---- figure: per-name de-noising trajectory grids ----
plt.rcParams["figure.facecolor"] = "white"
fig, axes = plt.subplots(2, 4, figsize=(15, 5.6))
axes = axes.ravel()
cmap = plt.get_cmap("tab20", vocab_size)
for j, (ax, traj, name) in enumerate(zip(axes, trajectories, final_names)):
    nt = len(traj)
    A = np.full((nt, max_len), 27)  # default underscore padding (id 27)
    for i, (_t, toks) in enumerate(traj):
        A[i] = toks.numpy()
    ax.imshow(A, aspect="auto", cmap=cmap, vmin=0, vmax=vocab_size - 1)
    ax.set_yticks(range(nt))
    ax.set_yticklabels([f"t={traj[i][0]:.1f}" for i in range(nt)], fontsize=8)
    ax.set_xticks([])
    ax.set_title(f"#{j}: {name}", fontsize=10)
fig.suptitle("FlowMore: parallel de-noising trajectory (t: 0.1 -> 1.0, 16 token positions)\n"
             + "color = token id, each row a discrete-flow-matching step",
             fontsize=11, y=1.02)
plt.tight_layout()
traj_fig = os.path.join(OUT, "name_sampling_trajectories.png")
plt.savefig(traj_fig, dpi=140, bbox_inches="tight")
plt.close(fig)
print("Saved", traj_fig)
print("\nGenerated names:")
print(final_names)
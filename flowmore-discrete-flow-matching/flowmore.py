import torch
import torch.nn as nn
from einops import rearrange

#set seed and device
torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#read in all words
words = open('names.txt', 'r').read().splitlines()
print(f"Sample names: {words[:8]}")

#build the vocabulary of characters and mappings to/from indices
chars     = sorted(list(set(''.join(words))))
stoi      = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
stoi['_'] = 27
itos      = {i:s for s,i in stoi.items()}

vocab_size= len(itos)
max_len = 16

#encode each name to fixed-lengh integer sequence
def encode(w, verbose = False):
    w = w + '.' + '_'* ( max_len - len(w) - 1)
    ids = [stoi[ch] for ch in w]
    if verbose == True:
        print(f"{w}: {ids}")
    return torch.tensor(ids).long()

#print samples    
for i in range(5):   
    encode(words[i], verbose = True)

#prepare data
data = torch.stack([encode(w) for w in words])

#model components
class MLP(nn.Module):
    """ Simple mlp with activations"""
    def __init__(self, dims, activation=torch.nn.SiLU()):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dims, 4*dims), nn.SiLU(),
            nn.Linear(4*dims, dims))

    def forward(self,x):
        return self.net(x)


class FourierEncoder(nn.Module):
    """Encodes a scalar timestep t into a sinusoidal embedding"""
    def __init__(self, dim):
        super().__init__()
        assert dim%2==0
        self.register_buffer('weights_f', torch.randn(1,dim//2))

    def forward(self,t):
        #t: [b]
        f     = 2 * torch.pi * self.weights_f * t.unsqueeze(-1)  #[b,dim/2]
        t_emb = torch.cat((torch.cos(f), torch.sin(f)), dim = 1) #[b,dim]
        return t_emb


class MHA(nn.Module):
    """Multihead self attention"""
    def __init__(self, dim, heads):
        super().__init__()
        assert dim%heads==0
        self.heads = heads
        self.scale = (dim//heads) ** 0.5
        self.qkv   = nn.Linear(dim, dim*3, bias = False)
        self.out_ff= nn.Linear(dim, dim, bias = False)

    def forward(self, x):
        #x: [b,n,d]
        Q,K,V = self.qkv(x).chunk(3, dim = -1)                       #3 x [b,n,d]
        hQ = rearrange(Q, 'b n (h d) -> (b h) n d', h = self.heads)  #[b*h, n, d/h]
        hK = rearrange(K, 'b n (h d) -> (b h) n d', h = self.heads)  #[b*h, n, d/h]
        hV = rearrange(V, 'b n (h d) -> (b h) n d', h = self.heads)  #[b*h, n, d/h]

        attn = torch.softmax(torch.einsum('bid,bjd->bij', hQ, hK)/self.scale, dim = -1) #[b*h, n, n]
        hZ   = torch.einsum('bij,bjd->bid', attn, hV)                                   #[b*h, n, d/h]
        Z    = rearrange(hZ, '(b h) n d -> b n (h d)', h = self.heads)                  #[b,n,d]
        return self.out_ff(Z)                                                           #[b,n,d]
        

class DiffusionTransformerLayer(nn.Module):
    """Diffusion Transformer Layer: Transformer block with adaptive layer norm modulation"""
    def __init__(self, dim, heads):
        super().__init__()

        self.norm1      = nn.LayerNorm(dim)
        self.attn       = MHA(dim,heads)
        self.norm2      = nn.LayerNorm(dim)
        self.ff         = MLP(dim)
        self.modulation = nn.Linear(dim,6*dim)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x, c):
        #x: [b,n,d], c: [b,d]
        g1, b1, a1, g2, b2, a2 = self.modulation(c).unsqueeze(1).chunk(6, dim = -1) #6 x [b,1,d]

        x = x + self.attn(self.norm1(x)* (1+g1) + b1) * a1    #[b,n,d]
        x = x + self.ff(  self.norm2(x)* (1+g2) + b2) * a2    #[b,n,d]
        return x

class DiffusionTransformer(nn.Module):
    """Diffusion transformer, which we use for discrete flow matching"""
    def __init__(self, num_layers, dim, heads, n_tokens, vocab_size):
        super().__init__()

        self.token_embedder = nn.Embedding(vocab_size, dim)
        self.pos_encoding   = nn.Parameter(torch.randn(n_tokens,dim))
        self.time_embedder  = FourierEncoder(dim)
    
        self.layer_list = nn.ModuleList([])
        for _ in range(num_layers):
            self.layer_list.append(DiffusionTransformerLayer(dim,heads))

        self.to_logits = nn.Linear(dim, vocab_size)

    def forward(self,x,t):
        #x: [b,n] #t: [b]
        x = self.token_embedder(x) + self.pos_encoding.unsqueeze(0) #[b,n,d]
        c = self.time_embedder(t)                                   #[b,d]

        for layer in self.layer_list:
            x = layer(x,c)       #[b,n,d]
        return self.to_logits(x) #[b,n,v]

#training
batch_size = 256

model = DiffusionTransformer(num_layers=3, dim=64, heads=4, n_tokens=max_len, vocab_size=vocab_size).to(device)
optim = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=10000, eta_min=1e-6)
loss_list = []

total = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {total:,}")

for step in range(10000):
    #sample a batch of names
    idx = torch.randint(0, len(data), (batch_size,))
    x_1 = data[idx]                                                         #[b,n] 
    
    #sample from source distribution - mask or uniform random
    #x_0 = torch.ones(x_1.shape) * stoi['_']                                #[b,n]
    x_0 = torch.randint(0,vocab_size,x_1.shape)                             #[b,n]

    #sample at time t
    t = torch.rand(batch_size)                                              #[b]
    x_t = torch.where(torch.rand(batch_size,max_len) < t[:,None], x_1, x_0) #[b,n]
    
    #predict probability that token xi at time t is the target token xi
    logits = model(x_t.long().to(device), t.to(device))                     #[b,n,v]
    loss = nn.functional.cross_entropy(
        logits.flatten(0,1),                                                #[b*n,v]
        x_1.flatten(0,1).squeeze().to(device)                               #[b*n]             
    )
    
    optim.zero_grad()
    loss.backward()
    optim.step()
    scheduler.step()
    
    loss_list.append(loss.item())
    if step%100==0:
        print(f"Step: {step}/10000, Loss: {loss.item():.4f}")


#sampling
model.eval()

for _ in range(10):
    #x_t = torch.ones(1,max_len).long()* stoi['_']
    x_t = torch.randint(0,vocab_size, (1,max_len)).long() #[1,n]
    t = 0.0

    while t < 1.0 - 1e-3:
        p1 = torch.softmax(model(x_t.to(device), torch.ones(1).to(device) * t), dim=-1).cpu()  #[1,n,v]
        h = min(0.1, 1.0 - t)                 
        one_hot_x_t = nn.functional.one_hot(x_t, vocab_size).float()                           #[1,n,v]
        u = (p1 - one_hot_x_t) / (1.0 - t)                                                     #[1,n,v]
        x_t = torch.distributions.Categorical(probs=one_hot_x_t + h * u).sample()              #[1,n]
        t += h
    
        print(f"t={t:.2f}: {''.join([itos[i] for i in x_t.squeeze().tolist()])}")

    print('\n')

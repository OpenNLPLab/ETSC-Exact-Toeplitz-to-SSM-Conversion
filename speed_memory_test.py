import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import time
from gpu_mem_track import MemTracker

device = "cuda:0"

class Rpe(nn.Module):
    def __init__(
        self,
        dim,
        outdim,
        residual,
        act="relu",
        bias=True,
        layers=3,
        norm_type="simplermsnorm",
    ):
        super().__init__()

        self.residual = residual
        self.outdim = outdim
        self.pos_dim = dim
        self.act = act
        self.pos_proj = nn.Linear(1, self.pos_dim, bias=bias)
        self.layers = nn.ModuleList([])
        for i in range(layers):
            self.layers.append(
                nn.Sequential(
                    nn.LayerNorm(self.pos_dim),
                    self.get_act(),
                    nn.Linear(self.pos_dim, self.pos_dim, bias=bias),
                )
            )
        self.out = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            self.get_act(),
            nn.Linear(self.pos_dim, self.outdim, bias=bias),
        )

    def get_act(self):
        return nn.ReLU(inplace=True)

    def forward(self, biases):
        x = self.pos_proj(biases)
        if self.residual:
            for m in self.layers:
                x = m(x) + x
        else:
            for m in self.layers:
                x = m(x)
        x = self.out(x)

        return x

class Tno(nn.Module):
    def __init__(
        self,
        h,
        dim,
        rpe_dim,
        causal=True,
        use_decay=True,
        use_multi_decay=False,
        residual=False,
        act="relu",
        par_type=1,
        gamma=0.999,
        bias=True,
        act_type="none",
        layers=3,
        norm_type="simplermsnorm",
        n=512,
    ):
        super().__init__()

        self.h = h
        self.dim = dim
        self.causal = causal
        self.par_type = par_type
        self.zero_value = 0
        self.use_decay = use_decay
        if self.use_decay:
            self.gamma = nn.Parameter(
                torch.ones(h, 1, dim) * gamma, requires_grad=False
            )
        self.use_multi_decay = use_multi_decay
        if self.use_multi_decay:
            self.lambda_ = gamma
            self.gamma = nn.Parameter(torch.randn(h, 1, dim))

        self.rpe = Rpe(
            dim=rpe_dim,
            outdim=h * dim,
            residual=residual,
            act=act,
            bias=bias,
            layers=layers,
            norm_type=norm_type,
        )

        self.act_fun = F.silu

        # for toeplitz
        self.max_seq = 0
        self.zero = torch.empty(0)
        self.pos = torch.empty(0)
        # for ssm
        self.lambda_ = None
        self.b = None
        self.n = n
        # for cache
        self.cache = None

    def get_pos(self, n):
        if self.par_type == 1:
            index = torch.arange(1, 1 + n).reshape(n, -1) * 1.0
        elif self.par_type == 2:
            index = torch.arange(1, 1 + n).reshape(n, -1) * 1.0 / n
        elif self.par_type == 3:
            index = torch.exp(torch.arange(1, 1 + n).reshape(n, -1) * 1.0 / n)

        return index

    def get_zero(self):
        index = torch.zeros(1).reshape(1, -1) * 1.0
        if self.par_type == 3:
            index = torch.exp(index)

        return index

    def get_neg(self, n):
        if self.causal:
            index = (
                torch.ones(self.h * n * self.dim).reshape(self.h, n, self.dim)
                * self.zero_value
            )
        else:
            if self.par_type == 1:
                index = -torch.arange(1, 1 + n).flip(0).reshape(n, -1) * 1.0
            elif self.par_type == 2:
                index = -torch.arange(1, 1 + n).flip(0).reshape(n, -1) * 1.0 / n

        return index

    def rpe_transform(self, x):
        # n, 1 -> n, (d * h)
        res = self.rpe(x)
        # n, (d * h) -> h, n, d
        res = rearrange(res, "n (h d) -> h n d", h=self.h)

        return res
    
    def forward_ssm(self, x, u=0):
        def transformation(t):
            # t: n, d
            t_pad = -torch.sum(t, dim=0, keepdim=True)
            t = torch.cat([t, t_pad], dim=0)
            m, d = t.shape
            # solve the eq
            t_ifft = torch.fft.ifft(t, n=m, dim=-2, norm="ortho")
            b = []
            for i in range(m):
                if i == 0:
                    b0 = torch.zeros(1, d).to(x.device)
                    b.append(b0)
                else:
                    b.append(t_ifft[i, None] / (np.sqrt(m)))
            b = torch.cat(b, dim=0)

            theta = -2j * np.pi * torch.arange(1, m).reshape(1, -1, 1).to(x.device) / m
            
            return theta, b

        gamma = rearrange(self.gamma, 'h n d -> n (h d)')
        # a0, a1, ... , a(n-1), a0, a(-(n-1)), ... , a(-1)
        if self.lambda_ == None:
            # 1, d, 1 -> h, 1, d
            zero, pos = self.get_cache_t(x, self.n)
            t = torch.cat([zero, pos], dim=1)
            t = rearrange(t, 'h n d -> n (h d)')
            theta, b = transformation(t)
            # 1, e, d
            self.lambda_ = torch.exp(theta).to(x.device) * gamma
            # e, d
            self.b = b[1:].to(x.device)

        output = []
        # x: b, h, n, d
        x = rearrange(x, 'b h n d -> b n (h d)')
        b, n, d = x.shape
        k = n // self.n
        # b, 1, d
        u = self.lambda_ * u + self.b * x
        # b, e, d -> b, 1, d
        output = torch.sum(u, dim=1, keepdim=True)
        output = rearrange(output, 'b n (h d) -> b h n d', h=self.h).real

        return output, u
    
    def forward_cache(self, x, prev_state=None, dim=-2):
        if prev_state != None:
            x = torch.cat([prev_state, x], dim=-2)
        # x: b, h, n, d
        n = x.shape[dim]
        ##### coef
        # 1, d, 1 -> h, 1, d
        zero, pos = self.get_cache_t(x, n)

        if self.use_decay or self.use_multi_decay:
            coef = torch.arange(1, n).reshape(1, -1, 1).to(x)
            if self.use_decay:
                gamma = self.gamma
            else:
                gamma = torch.sigmoid(self.gamma)
                gamma = self.lambda_ + (1 - self.lambda_) * gamma
            gamma = gamma ** coef
            pos = gamma * pos
        a = torch.cat([zero, pos], dim=1)
        a = self.act_fun(a)

        # x: b, h, n, d
        # a: h, l, d
        output = torch.einsum('h n d, b h n d -> b h d', a, x).unsqueeze(1)
        prev_state = x

        return output, prev_state

    def forward_fft(self, x, dim=-2):
        # x: b, h, n, d
        n = x.shape[dim]
        # a0, a1, ... , a(n-1), a0, a(-(n-1)), ... , a(-1)
        ##### coef
        # 1, d, 1 -> h, 1, d
        zero, pos = self.get_cache_t(x, n)
        # m = max(n, 2)
        # zero = self.rpe_transform(self.get_zero().to(x))
        # pos = self.rpe_transform(self.get_pos(m - 1).to(x)[:n - 1, ...])

        if self.use_decay or self.use_multi_decay:
            coef = torch.arange(1, n).reshape(1, -1, 1).to(x)
            if self.use_decay:
                gamma = self.gamma
            else:
                gamma = torch.sigmoid(self.gamma)
                gamma = self.lambda_ + (1 - self.lambda_) * gamma
            gamma = gamma ** coef
            pos = gamma * pos
        a = torch.cat([zero, pos, zero], dim=1)
        a = self.act_fun(a)

        # x: b, h, n, d
        # a: h, l, d
        output = self.compute(x, a, dim, n)

        return output
    
    def compute(self, x, a, dim, n):
        # x: b, h, n, d
        # a: h, n, d
        y = torch.fft.rfft(x, 2 * n, dim=dim)
        v = torch.fft.rfft(a, 2 * n, dim=dim).unsqueeze(0)
        u = v * y
        output = torch.fft.irfft(u, 2 * n, dim=dim)[:, :, :n, :]

        return output
    
    def get_cache_t(self, x, n):
        # first time
        if self.max_seq == 0:
            self.max_seq = self.n
            zero = self.rpe_transform(self.get_zero().to(x))
            pos = self.rpe_transform(self.get_pos(self.n - 1).to(x))
            self.zero = zero
            self.pos = pos

        if n > self.max_seq:
            self.max_seq = 2 * n
            zero = self.rpe_transform(self.get_zero().to(x))
            pos = self.rpe_transform(self.get_pos(self.max_seq - 1).to(x))
            self.zero = zero
            self.pos = pos
        
        return self.zero, self.pos[:, :n - 1]
    
class TnnLm(nn.Module):
    def __init__(
        self,
        h=1,
        dim=512,
        rpe_dim=64,
        causal=True,
        n_layers=12,
        vocab_size=50000,
        n=512,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = torch.nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                Tno(
                    h=h,
                    dim=dim,
                    rpe_dim=rpe_dim,
                    n=n,
                    causal=causal
                )
            )
        self.lm_head = nn.Linear(dim, vocab_size)
        self.n_layers = n_layers
    
    def forward_ssm(self, x, u_states=[]):
        # x: b, n, -> b, h, n, d
        feature = self.embedding(x).unsqueeze(1)
        if len(u_states) == 0:
            u_states = [0] * self.n_layers
        new_u_states = []
        for i, layer in enumerate(self.layers):
            feature, u = layer.forward_ssm(feature, u_states[i])
            new_u_states.append(u)
        # b, h, n, d -> b, n, d
        logits = self.lm_head(feature).squeeze(1)

        return logits, new_u_states
    
    def forward_cache(self, x, prev_states=[]):
        # x: b, n, d -> b, h, n, d
        feature = self.embedding(x).unsqueeze(1)
        if len(prev_states) == 0:
            prev_states = [None] * self.n_layers
        new_prev_states = []
        for i, layer in enumerate(self.layers):
            feature, prev_state = layer.forward_cache(feature, prev_states[i])
            new_prev_states.append(prev_state)
        # b, h, n, d -> b, n, d
        logits = self.lm_head(feature).squeeze(1)

        return logits, new_prev_states
    
    def forward_fft(self, x):
        # x: b, n, d -> b, h, n, d
        feature = self.embedding(x).unsqueeze(1)
        for layer in self.layers:
            feature = layer.forward_fft(feature)
        # b, h, n, d -> b, n, d
        logits = self.lm_head(feature).squeeze(1)

        return logits
    
def get_memory():
    mb_used = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    torch.cuda.reset_peak_memory_stats(device)

    return mb_used

def speed_test(n, h, d, layer=12, m=1, name="layer"):
    model = TnnLm(h=h, dim=d, n_layers=layer).to(device)
    model.eval()
    name = f"{name}_{layer}_seqlen_{n}_dim_{d}"
    gpu_tracker = MemTracker(name=name)

    torch.cuda.reset_peak_memory_stats(device)
    start = time.time()
    for _ in range(m):
        x = torch.ones(1, 1).long().to(device)
        u_states = []
        for i in range(n):
            logits, u_states = model.forward_ssm(x, u_states)
            logits = logits[:, -1, :]
            x = torch.argmax(logits, dim=-1).unsqueeze(-1)
    end = time.time()
    t1 = (end - start) / n / m
    m1 = get_memory()
    torch.cuda.empty_cache()
    gpu_tracker.track()

    torch.cuda.reset_peak_memory_stats(device)
    start = time.time()
    for _ in range(m):
        x = torch.ones(1, 1).long().to(device)
        prev_states = []
        for i in range(n):
            logits, prev_states = model.forward_cache(x, prev_states)
            logits = logits[:, -1, :]
            x = torch.argmax(logits, dim=-1).unsqueeze(-1)

    end = time.time()
    t2 = (end - start) / n / m
    m2 = get_memory()
    torch.cuda.empty_cache()
    gpu_tracker.track()

    torch.cuda.reset_peak_memory_stats(device)
    start = time.time()
    for _ in range(m):
        x = torch.ones(1, 1).long().to(device)
        start = time.time()
        for i in range(n):
            logits = model.forward_fft(x)[:, -1, :]
            token = torch.argmax(logits, dim=-1).unsqueeze(-1)
            x = torch.cat([x, token], dim=1)
    end = time.time()
    t3 = (end - start) / n / m
    m3 = get_memory()
    torch.cuda.empty_cache()
    gpu_tracker.track()

    print(f"{layer}, {n}, {d}, {t1}, {t2}, {t3}, {m1}, {m2}, {m3}")

    return t1, t2, t3

n = 512
h = 1
d = 512
b = 1
m = 1
k = 15

# start = time.time()
# # seqlen vs speed
# ns = [64,  128,  256,  512, 1024, 2048, 4096, 8192]
# d = 64
# layer = 2
# n_vs_speed = {
#     "ssm": [],
#     "tnn_naive": [],
#     "tnn_fft_naive": [],
# }
# t = torch.rand(ns[-1], d).to(device) * 10
# print("seqlen vs speed")
# print("nlayers, seqlen, feature_dim, ssm, tnn_cache, tnn_fft_naive, ssm, tnn_cache, tnn_fft_naive")
# for n in ns:
#     t1, t2, t3 = speed_test(n, h, d, layer=layer, m=m, name="seqlen")
#     n_vs_speed["ssm"].append(t1)
#     n_vs_speed["tnn_naive"].append(t2)
#     n_vs_speed["tnn_fft_naive"].append(t3)

# # feature_dim vs speed
# ds = [64,  128,  192,  256,  320,  384,  448,  512,  576,  640,  704, 768,  832,  896,  960, 1024, 1088, 1152, 1216, 1280]
# n = 2048
# layer = 2
# d_vs_speed = {
#     "ssm": [],
#     "tnn_naive": [],
#     "tnn_fft_naive": [],
# }
# print("feature_dim vs speed")
# print("nlayers, seqlen, feature_dim, ssm, tnn_cache, tnn_fft_naive, ssm, tnn_cache, tnn_fft_naive")
# for d in ds:
#     t1, t2, t3 = speed_test(n, h, d, layer=layer, m=m, name="feature_dim")
#     d_vs_speed["ssm"].append(t1)
#     d_vs_speed["tnn_naive"].append(t2)
#     d_vs_speed["tnn_fft_naive"].append(t3)

# layer vs speed
layers = np.arange(1, 13)
n = 2048
d = 64
d_vs_speed = {
    "ssm": [],
    "tnn_naive": [],
    "tnn_fft_naive": [],
}
print("nlayers vs speed")
print("nlayers, seqlen, feature_dim, ssm, tnn_cache, tnn_fft_naive, ssm, tnn_cache, tnn_fft_naive")
for layer in layers:
    t1, t2, t3 = speed_test(n, h, d, layer, m, name="layer")
    d_vs_speed["ssm"].append(t1)
    d_vs_speed["tnn_naive"].append(t2)
    d_vs_speed["tnn_fft_naive"].append(t3)
end = time.time()
print(end - start)
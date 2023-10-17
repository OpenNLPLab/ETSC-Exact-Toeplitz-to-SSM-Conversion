# Exact-Toeplitz-to-SSM-Conversion
Official implementation of the algorithm ETSC: [Exact Toeplitz-to-SSM Conversion] our EMNLP 2023 paper - Accelerating Toeplitz Neural Network with Constant-time Inference Complexity.

Paper coming soon.

## Experiments
`transformation.py` for Algorithm 1. `speed_test.py` for speed memory test.

## Inference
If you want to use the algorithm during inference stage, please take the follow code as reference:

```python
def forward_ssm(self, x, dim=-2, normalize=False):
    def transformation(t):
        # t: n, d
        t_pad = -torch.sum(t, dim=0, keepdim=True)
        t = torch.cat([t, t_pad], dim=0)
        m, d = t.shape
        # solve the eq
        t_ifft = torch.fft.ifft(t, n=m, dim=-2, norm="ortho")
        b = torch.cat([torch.zeros(1, d).to(t.device), t_ifft[1:m] / np.sqrt(m)], dim=0)

        theta = -2j * np.pi * torch.arange(1, m).reshape(1, -1, 1).to(t.device) / m
        
        return theta, b

    gamma = rearrange(self.gamma, 'h n d -> n (h d)')
    # a0, a1, ... , a(n-1), a0, a(-(n-1)), ... , a(-1)
    if self.lambda_ == None:
        # 1, d, 1 -> h, 1, d
        zero = self.rpe_transform(self.get_zero().to(x))
        pos = self.rpe_transform(self.get_pos(self.n - 1).to(x))
        t = torch.cat([zero, pos], dim=1)
        t = rearrange(t, 'h n d -> n (h d)')
        theta, b = transformation(t)
        # 1, e, d
        self.lambda_ = torch.exp(theta).to(t.device) * gamma
        # e, d
        self.b = b[1:].to(x.device)

    output = []
    # x: b, h, n, d
    x = rearrange(x, 'b h n d -> b n (h d)')
    b, n, d = x.shape
    k = n // self.n
    # b, 1, d
    zero = torch.zeros(b, 1, d).to(x.device)
    u = zero
    for i in range(n):
        u = self.lambda_ * u + self.b * x[:, i]
        # b, e, d -> b, 1, d
        y = torch.sum(u, dim=1, keepdim=True)
        output.append(y)

    output = torch.cat(output, dim=1)
    output = rearrange(output, 'b n (h d) -> b h n d', h=self.h).real

    return output
```
The parameters are init as follows:
```python
self.lambda_ = None
self.b = None
self.n = training length
self.forward = self.forward_ssm
```

The origin inference code is as follows:
```python
def forward_causal(self, x, dim=-2, normalize=False):
    # x: b, h, n, d
    n = x.shape[dim]
    # a0, a1, ... , a(n-1), a0, a(-(n-1)), ... , a(-1)
    ##### coef
    # 1, d, 1 -> h, 1, d
    zero = self.rpe_transform(self.get_zero().to(x))
    pos = self.rpe_transform(self.get_pos(n - 1).to(x))

    if self.use_decay or self.use_multi_decay:
        coef = torch.arange(1, n).reshape(1, -1, 1).to(x)
        if self.use_decay:
            gamma = self.gamma
        else:
            gamma = torch.sigmoid(self.gamma)
            gamma = self.lambda_ + (1 - self.lambda_) * gamma
        gamma = gamma**coef
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
    y = torch.fft.rfft(x.float(), 2 * n, dim=dim)
    v = torch.fft.rfft(a.float(), 2 * n, dim=dim).unsqueeze(0)
    u = v * y
    output = torch.fft.irfft(u, 2 * n, dim=dim)[:, :, :n, :].to(x.dtype)

    return output
```
import torch
import numpy as np
import time

def transformation(t):
    # t: n, d
    t_pad = -torch.sum(t, dim=0, keepdim=True)
    t = torch.cat([t, t_pad], dim=0)
    m, d = t.shape
    # solve the eq
    t_ifft = torch.fft.ifft(t, n=m, dim=-2, norm="ortho")
    b = torch.cat([torch.zeros(1, d).to(t.device), t_ifft[1:m] / np.sqrt(m)], dim=0)

    theta = -2j * torch.pi * torch.arange(1, m).reshape(1, -1, 1) / m
    theta = theta.to(t.device)
    
    return theta, b

def verify(t, b):
    # t: n, d
    t_pad = -torch.sum(t, dim=0, keepdim=True)
    t = torch.cat([t, t_pad], dim=0)
    m, d = t.shape
    t_res = np.sqrt(m) * (torch.fft.fft(b, n=m, dim=-2, norm="ortho"))

    print(torch.norm(t - t_res))

n = 512
d = 256
t = torch.rand(n, d) * 10
theta, b = transformation(t)
verify(t, b)

# seqlen vs error
log_ns = np.arange(6, 15)
d = 64
ns = 2 ** log_ns
print(d, list(ns))
error = []
times = []

# n vs error
error = []
times = []
for log_n in log_ns:
    n = 2 ** log_n
    t = torch.rand(n, d).cuda() * 10
    start = time.time()
    theta, b = transformation(t)
    end = time.time()
    kernel = compute(theta, b, n)
    e = torch.norm((kernel - t)) / torch.norm(t) * 100
    error.append(e.item())
    times.append(end - start)
print("Gpu Version")
print("Time", times)
print("Relative Error", error)

print("Feature dim")
# feature dim vs error
n = 2048
log_ds = np.arange(6, 15)
ds = 2 ** log_ds
print(n, list(ds))
error = []
times = []
for log_d in log_ds:
    d = 2 ** log_d
    t = torch.rand(n, d).cuda() * 10
    start = time.time()
    theta, b = transformation(t)
    end = time.time()
    kernel = compute(theta, b, n)
    e = torch.norm((kernel - t)) / torch.norm(t) * 100
    error.append(e.item())
    times.append(end - start)
print("Gpu Version")
print("Time", times)
print("Relative Error", error)

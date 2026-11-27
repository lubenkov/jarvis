import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, head_dim, theta=10000.0, max_seq_len=4096):
        super().__init__()
        assert head_dim % 2 == 0
        self.head_dim = head_dim
        self.theta = theta

        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        device = self.cos_cached.device
        inv_freq = 1.0 / (self.theta ** (
            torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim))
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.cos_cached = freqs.cos()
        self.sin_cached = freqs.sin()

    def _ensure_cache(self, seq_len):
        cached = self.cos_cached.shape[0]
        if seq_len <= cached:
            return
        self._build_cache(max(seq_len, cached * 2))

    def rotate(self, q, k, offset=0):
        seq_len = q.shape[2]
        self._ensure_cache(offset + seq_len)
        cos = self.cos_cached[offset:offset + seq_len].to(q.dtype)[None, None]
        sin = self.sin_cached[offset:offset + seq_len].to(q.dtype)[None, None]

        q_even, q_odd = q[..., 0::2], q[..., 1::2]
        k_even, k_odd = k[..., 0::2], k[..., 1::2]

        q_rot = torch.stack((q_even * cos - q_odd * sin, q_even * sin + q_odd * cos), dim=-1)
        k_rot = torch.stack((k_even * cos - k_odd * sin, k_even * sin + k_odd * cos), dim=-1)
        return q_rot.flatten(-2), k_rot.flatten(-2)

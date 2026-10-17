import torch
import torch.nn as nn


class RoPE(nn.Module):
    """Rotary Position Embeddings.

    Позиция кодируется поворотом пар соседних компонент q и k на угол,
    зависящий от номера позиции. Позволяет обходиться без обучаемых эмбеддингов позиций.
    """

    def __init__(self, head_dim, theta=10000.0, max_seq_len=4096):
        super().__init__()
        assert head_dim % 2 == 0

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)  # (max_seq_len, head_dim // 2)

        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def rotate(self, q, k, offset=0):
        """Поворачивает q и k. offset — позиция первого токена (для генерации с KV-кэшем).

        Метод намеренно не называется apply: он перекрыл бы nn.Module.apply,
        и рекурсивный обход модулей (например, инициализация весов через
        model.apply(fn)) падал бы с TypeError.
        """
        seq_len = q.shape[2]
        cos = self.cos_cached[offset:offset + seq_len].to(q.dtype)[None, None]
        sin = self.sin_cached[offset:offset + seq_len].to(q.dtype)[None, None]

        q_even, q_odd = q[..., 0::2], q[..., 1::2]
        k_even, k_odd = k[..., 0::2], k[..., 1::2]

        q_rot = torch.stack((q_even * cos - q_odd * sin, q_even * sin + q_odd * cos), dim=-1)
        k_rot = torch.stack((k_even * cos - k_odd * sin, k_even * sin + k_odd * cos), dim=-1)
        return q_rot.flatten(-2), k_rot.flatten(-2)

"""Exact gate from the 2026-10-04 implementation specification."""
import torch
from torch import nn
from torch.nn import functional as F


class CoarseContextConcat(nn.Module):
    def __init__(self, c3=64, c4=128, hidden=32, mode="p3p4"):
        super().__init__()
        if mode not in {"p3p4", "p3_only"}:
            raise ValueError("Unknown conditioning mode")
        self.c3, self.c4, self.mode = c3, c4, mode
        self.fuse = nn.Conv2d(c3 + c4, hidden, 1, bias=True)
        self.act = nn.SiLU(inplace=False)
        self.logits = nn.Conv2d(hidden, 1, 1, bias=True)
        nn.init.zeros_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)

    def gate(self, t3, t4, target_size):
        if t3.shape[1] != self.c3 or t4.shape[1] != self.c4:
            raise ValueError("Unexpected feature channels")
        t4_at_3 = F.interpolate(t4, size=t3.shape[-2:], mode="nearest")
        if self.mode == "p3_only":
            t4_at_3 = torch.zeros_like(t4_at_3)
        hidden = self.act(self.fuse(torch.cat((t3, t4_at_3), dim=1)))
        g3 = 2.0 * torch.sigmoid(self.logits(hidden))
        return F.interpolate(g3, size=target_size, mode="nearest")

    def forward(self, features):
        u3, t3, t4, b2 = features
        if u3.shape[1] != self.c3 or u3.shape[-2:] != b2.shape[-2:]:
            raise ValueError("Incoming P3 and lateral P2 do not match")
        g2 = self.gate(t3, t4, u3.shape[-2:])
        return torch.cat((u3 * g2, b2), dim=1)

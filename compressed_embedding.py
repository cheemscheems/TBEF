import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def tsvd_decompose(weight, rank):
    """Return low-rank factors A and B for weight ~= A @ B."""
    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
    u = u[:, :rank]
    s = s[:rank]
    vh = vh[:rank, :]
    a = u * s.unsqueeze(0)
    b = vh
    rel_error = torch.norm(weight - a @ b).item() / torch.norm(weight).item()
    return a, b, rel_error


class GatedTSVDEmbedding(nn.Module):
    """Global TSVD embedding with a learned per-token rank gate."""

    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = None
        self.projection = None
        self.gate = None
        self.gate_sparsity_lambda = 0.0
        self.is_decomposed = False

    def decompose(self, weight, rank, gate_init_bias=2.197, gate_sparsity_lambda=1e-4):
        device = weight.device
        dtype = weight.dtype
        a, b, rel_error = tsvd_decompose(weight.float(), rank)
        logger.info("GatedTSVD: rank=%d, rel_error=%.6f", rank, rel_error)

        self.embedding = nn.Embedding(self.vocab_size, rank)
        self.embedding.weight.data = a.to(device=device, dtype=dtype)

        self.projection = nn.Linear(rank, self.hidden_size, bias=False)
        self.projection.weight.data = b.t().to(device=device, dtype=dtype)

        self.gate = nn.Embedding(self.vocab_size, rank)
        nn.init.constant_(self.gate.weight, gate_init_bias)
        self.gate_sparsity_lambda = gate_sparsity_lambda
        self.is_decomposed = True

        baseline_params = self.vocab_size * self.hidden_size
        gated_params = self.vocab_size * rank * 2 + rank * self.hidden_size
        logger.info(
            "GatedTSVD params: baseline=%d, gated=%d, ratio=%.4f",
            baseline_params,
            gated_params,
            gated_params / baseline_params,
        )

    def forward(self, input_ids):
        low_rank = self.embedding(input_ids)
        gate = torch.sigmoid(self.gate(input_ids))
        return self.projection(low_rank * gate)

    def get_extra_loss(self):
        if self.gate_sparsity_lambda <= 0:
            return 0.0
        return self.gate_sparsity_lambda * torch.mean(torch.sigmoid(self.gate.weight))


def create_compressed_embedding(method, vocab_size, hidden_size):
    if method != "gate":
        raise ValueError("TBEF only supports compression_method='gate'.")
    return GatedTSVDEmbedding(vocab_size, hidden_size)

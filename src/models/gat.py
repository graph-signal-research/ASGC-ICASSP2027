from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DenseGAT(nn.Module):
    """Dense directed GAT used for METR-LA.

    Inputs are `x [B,N,D]`, `observed [B,N,N]`, and `mask [B,N,N]`.
    Matrix entry `A[j,i]` denotes edge `j -> i`; target nodes aggregate
    over the source index. The output has shape `[B,N,N]`.
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden: int = 12,
        heads: int = 2,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.hidden = hidden
        self.linear = nn.Linear(input_dim, heads * hidden, bias=False)
        self.attn_source = nn.Parameter(torch.empty(heads, hidden))
        self.attn_target = nn.Parameter(torch.empty(heads, hidden))
        self.edge_scale = nn.Parameter(torch.ones(heads))
        self.decoder = nn.Sequential(
            nn.Linear(4 * heads * hidden, 32),
            nn.ELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attn_source)
        nn.init.xavier_uniform_(self.attn_target)

    def forward(
        self,
        x: torch.Tensor,
        observed: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, node_count, _ = x.shape
        transformed = self.linear(x).view(
            batch_size,
            node_count,
            self.heads,
            self.hidden,
        )

        source_score = torch.einsum(
            'bnhd,hd->bnh', transformed, self.attn_source
        )
        target_score = torch.einsum(
            'bnhd,hd->bnh', transformed, self.attn_target
        )
        scores = F.leaky_relu(
            source_score[:, :, None, :]
            + target_score[:, None, :, :]
            + observed[:, :, :, None] * self.edge_scale,
            0.2,
        )

        diagonal = torch.eye(
            node_count,
            dtype=torch.bool,
            device=x.device,
        )[None, :, :]
        allowed = mask.clone() | diagonal
        scores = scores.masked_fill(
            ~allowed[:, :, :, None],
            -1.0e9,
        )
        attention = torch.softmax(scores, dim=1)

        aggregated = torch.einsum(
            'bijh,bihd->bjhd', attention, transformed
        ).reshape(
            batch_size,
            node_count,
            self.heads * self.hidden,
        )
        aggregated = F.elu(aggregated)

        source = aggregated[:, :, None, :].expand(
            -1, -1, node_count, -1
        )
        target = aggregated[:, None, :, :].expand(
            -1, node_count, -1, -1
        )
        pair = torch.cat(
            [source, target, torch.abs(source - target), source * target],
            dim=-1,
        )
        prediction = self.decoder(pair).squeeze(-1)
        return prediction.masked_fill(diagonal, 0.0)


def _grouped_softmax_by_target(
    scores: torch.Tensor,
    target_index: torch.Tensor,
    node_count: int,
) -> torch.Tensor:
    if scores.shape[0] == 0:
        return scores

    expanded_index = target_index[:, None].expand(-1, scores.shape[1])
    maxima = torch.full(
        (node_count, scores.shape[1]),
        -torch.inf,
        dtype=scores.dtype,
        device=scores.device,
    )
    maxima.scatter_reduce_(
        0,
        expanded_index,
        scores,
        reduce='amax',
        include_self=True,
    )

    stable = scores - maxima[target_index]
    exponential = torch.exp(stable)
    denominator = torch.zeros_like(maxima)
    denominator.scatter_add_(0, expanded_index, exponential)
    return exponential / denominator[target_index].clamp_min(
        torch.finfo(scores.dtype).tiny
    )


class DirectedGATLayer(nn.Module):
    """Sparse directed attention layer for the synthetic GAT."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        head_count: int = 1,
        concat_heads: bool = True,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_count = head_count
        self.concat_heads = concat_heads
        self.dropout = dropout
        self.negative_slope = negative_slope

        self.linear = nn.Linear(
            input_dim,
            head_count * output_dim,
            bias=False,
        )
        self.attention_source = nn.Parameter(
            torch.empty(head_count, output_dim)
        )
        self.attention_target = nn.Parameter(
            torch.empty(head_count, output_dim)
        )
        combined_dim = head_count * output_dim if concat_heads else output_dim
        self.residual = nn.Linear(input_dim, combined_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(combined_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attention_source)
        nn.init.xavier_uniform_(self.attention_target)
        nn.init.xavier_uniform_(self.residual.weight)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        node_count = node_features.shape[0]
        source, target = edge_index
        transformed = self.linear(node_features).view(
            node_count,
            self.head_count,
            self.output_dim,
        )

        if edge_index.shape[1] == 0:
            aggregated = torch.zeros_like(transformed)
        else:
            score = F.leaky_relu(
                (transformed[source] * self.attention_source).sum(-1)
                + (transformed[target] * self.attention_target).sum(-1),
                negative_slope=self.negative_slope,
            )
            coefficient = F.dropout(
                _grouped_softmax_by_target(score, target, node_count),
                p=self.dropout,
                training=self.training,
            )
            message = coefficient.unsqueeze(-1) * transformed[source]
            aggregated = torch.zeros_like(transformed)
            aggregated.scatter_add_(
                0,
                target[:, None, None].expand_as(message),
                message,
            )

        if self.concat_heads:
            output = aggregated.reshape(node_count, -1)
        else:
            output = aggregated.mean(1)
        return output + self.residual(node_features) + self.bias


class DirectedGATPointModel(nn.Module):
    """Two-layer point-GAT used for preliminary synthetic edge completion."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 8,
        embedding_dim: int = 8,
        decoder_hidden_dim: int = 8,
        head_count: int = 1,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.layer1 = DirectedGATLayer(
            input_dim,
            hidden_dim,
            head_count,
            True,
            dropout,
            negative_slope,
        )
        self.layer2 = DirectedGATLayer(
            hidden_dim * head_count,
            embedding_dim,
            1,
            False,
            dropout,
            negative_slope,
        )
        self.dropout = dropout
        self.decoder = nn.Sequential(
            nn.Linear(2 * embedding_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, 1),
        )

    def encode(
        self,
        node_features: torch.Tensor,
        fit_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        hidden = F.elu(self.layer1(node_features, fit_edge_index))
        hidden = F.dropout(
            hidden,
            p=self.dropout,
            training=self.training,
        )
        return self.layer2(hidden, fit_edge_index)

    def forward(
        self,
        node_features: torch.Tensor,
        fit_edge_index: torch.Tensor,
        prediction_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.encode(node_features, fit_edge_index)
        source, target = prediction_edge_index
        pairs = torch.cat(
            [embedding[source], embedding[target]],
            dim=-1,
        )
        return torch.sigmoid(self.decoder(pairs).squeeze(-1))

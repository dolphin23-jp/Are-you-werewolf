"""PyTorch player/event Transformer implementing the learned-policy contract.

The environment and encoder remain framework agnostic. This module consumes the
same information-safe :class:`EncodedPolicyObservation` used by the NumPy smoke
model, but preserves the token structure instead of flattening it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from app.engine.roles import RoleName
from app.training.actions import (
    ActionType,
    Channel,
    ResultValue,
    Scope,
    Stance,
    Topic,
)
from app.training.encoding import (
    MAX_DAWN_EVENTS,
    MAX_SEATS,
    MAX_SEMANTIC_EVENTS,
    MAX_VOTE_EVENTS,
    EncodedPolicyObservation,
)
from app.training.policy_contract import PolicyHeadSizes, PolicyLogits

_GLOBAL_VOCABS = (32, 16, 256, 16, MAX_SEATS + 1, MAX_SEATS + 1, len(RoleName) + 1, 2)
_PLAYER_VOCABS = (
    MAX_SEATS + 1,
    2,
    2,
    32,
    4,
    len(RoleName) + 1,
    2,
    3,
    3,
    32,
    32,
)
_SEMANTIC_VOCABS = (
    32,
    256,
    MAX_SEATS + 1,
    len(Channel) + 1,
    len(ActionType) + 1,
    len(Topic) + 1,
    MAX_SEATS + 1,
    MAX_SEATS + 1,
    len(RoleName) + 1,
    len(ResultValue) + 1,
    8,
    32,
    len(Scope) + 1,
    len(Stance) + 1,
)
_VOTE_VOCABS = (32, 16, MAX_SEATS + 1, MAX_SEATS + 1)
_DAWN_VOCABS = (32, 2, MAX_SEATS + 1, MAX_SEATS + 1)
_SEQUENCE_LENGTH = 2 + MAX_SEATS + MAX_SEMANTIC_EVENTS + MAX_VOTE_EVENTS + MAX_DAWN_EVENTS
_PLAYER_START = 2
_SEMANTIC_START = _PLAYER_START + MAX_SEATS


@dataclass(frozen=True)
class TransformerPolicyConfig:
    d_model: int = 96
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 384
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.nhead <= 0 or self.d_model % self.nhead != 0:
            raise ValueError("nhead must divide d_model")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.dim_feedforward <= 0:
            raise ValueError("dim_feedforward must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class TorchPolicyTensorOutput:
    """Batched differentiable policy/value outputs before legal masking."""

    timing: Tensor
    action_type: Tensor
    topic: Tensor
    target: Tensor
    secondary_target: Tensor
    role: Tensor
    result: Tensor
    quantity: Tensor
    referenced_day: Tensor
    scope: Tensor
    stance: Tensor
    reference_event: Tensor
    vote_target: Tensor
    night_topic: Tensor
    night_target: Tensor
    value: Tensor

    def head(self, name: str) -> Tensor:
        value = getattr(self, name, None)
        if not isinstance(value, Tensor):
            raise KeyError(name)
        return value


class _FeatureTokenEncoder(nn.Module):
    """Embed heterogeneous integer fields without imposing ordinal semantics."""

    def __init__(self, vocab_sizes: tuple[int, ...], d_model: int) -> None:
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.embeddings = nn.ModuleList(
            nn.Embedding(vocab_size, d_model, padding_idx=0)
            for vocab_size in vocab_sizes
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, fields: Tensor) -> Tensor:
        if fields.shape[-1] != len(self.embeddings):
            raise ValueError("unexpected encoded token width")
        result: Tensor | None = None
        for index, (embedding, vocab_size) in enumerate(
            zip(self.embeddings, self.vocab_sizes, strict=True)
        ):
            # Large day/tick/count values share a final overflow bucket. This is
            # model-side bucketing only; the information-safe source observation
            # and recorded trajectory remain unchanged.
            values = fields[..., index].clamp(min=0, max=vocab_size - 1)
            embedded = embedding(values)
            result = embedded if result is None else result + embedded
        if result is None:
            raise RuntimeError("feature token encoder has no fields")
        normalized: Tensor = self.norm(result)
        return normalized


class _PointerHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.scale = math.sqrt(d_model)

    def forward(self, context: Tensor, candidates: Tensor) -> Tensor:
        query = self.query(context).unsqueeze(1)
        keys = self.key(candidates)
        return torch.sum(query * keys, dim=-1) / self.scale


class TorchTransformerPolicy(nn.Module):
    """Token-aware policy/value model with player and event pointer heads."""

    def __init__(
        self,
        config: TransformerPolicyConfig | None = None,
        *,
        sizes: PolicyHeadSizes | None = None,
    ) -> None:
        super().__init__()
        self.config = config or TransformerPolicyConfig()
        self.sizes = sizes or PolicyHeadSizes()
        d_model = self.config.d_model

        self.global_encoder = _FeatureTokenEncoder(_GLOBAL_VOCABS, d_model)
        self.player_encoder = _FeatureTokenEncoder(_PLAYER_VOCABS, d_model)
        self.semantic_encoder = _FeatureTokenEncoder(_SEMANTIC_VOCABS, d_model)
        self.vote_encoder = _FeatureTokenEncoder(_VOTE_VOCABS, d_model)
        self.dawn_encoder = _FeatureTokenEncoder(_DAWN_VOCABS, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position_embedding = nn.Embedding(_SEQUENCE_LENGTH, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=self.config.num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        self.timing_head = nn.Linear(d_model, self.sizes.timing)
        self.action_type_head = nn.Linear(d_model, self.sizes.action_type)
        self.topic_head = nn.Linear(d_model, self.sizes.topic)
        self.role_head = nn.Linear(d_model, self.sizes.role)
        self.result_head = nn.Linear(d_model, self.sizes.result)
        self.quantity_head = nn.Linear(d_model, self.sizes.quantity)
        self.referenced_day_head = nn.Linear(d_model, self.sizes.referenced_day)
        self.scope_head = nn.Linear(d_model, self.sizes.scope)
        self.stance_head = nn.Linear(d_model, self.sizes.stance)
        self.night_topic_head = nn.Linear(d_model, self.sizes.night_topic)
        self.value_head = nn.Linear(d_model, 1)

        self.target_pointer = _PointerHead(d_model)
        self.secondary_target_pointer = _PointerHead(d_model)
        self.vote_target_pointer = _PointerHead(d_model)
        self.night_target_pointer = _PointerHead(d_model)
        self.reference_event_pointer = _PointerHead(d_model)

        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    @property
    def device(self) -> torch.device:
        return self.cls_token.device

    def forward(self, observation: EncodedPolicyObservation) -> PolicyLogits:
        """Framework-agnostic inference adapter expected by rollout policies."""
        with torch.no_grad():
            output = self.forward_batch((observation,))
        logits = PolicyLogits(
            timing=_as_tuple(output.timing[0]),
            action_type=_as_tuple(output.action_type[0]),
            topic=_as_tuple(output.topic[0]),
            target=_as_tuple(output.target[0]),
            secondary_target=_as_tuple(output.secondary_target[0]),
            role=_as_tuple(output.role[0]),
            result=_as_tuple(output.result[0]),
            quantity=_as_tuple(output.quantity[0]),
            referenced_day=_as_tuple(output.referenced_day[0]),
            scope=_as_tuple(output.scope[0]),
            stance=_as_tuple(output.stance[0]),
            reference_event=_as_tuple(output.reference_event[0]),
            vote_target=_as_tuple(output.vote_target[0]),
            night_topic=_as_tuple(output.night_topic[0]),
            night_target=_as_tuple(output.night_target[0]),
            value=float(output.value[0].detach().cpu()),
        )
        logits.validate(self.sizes)
        return logits

    def forward_batch(
        self,
        observations: Sequence[EncodedPolicyObservation],
    ) -> TorchPolicyTensorOutput:
        """Differentiable batched forward pass used by the Torch trainer."""
        if not observations:
            raise ValueError("transformer forward requires at least one observation")
        device = self.device
        global_fields = torch.tensor(
            [observation.global_features for observation in observations],
            dtype=torch.long,
            device=device,
        )
        player_fields = torch.tensor(
            [observation.player_tokens for observation in observations],
            dtype=torch.long,
            device=device,
        )
        semantic_fields = torch.tensor(
            [observation.semantic_tokens for observation in observations],
            dtype=torch.long,
            device=device,
        )
        vote_fields = torch.tensor(
            [observation.vote_tokens for observation in observations],
            dtype=torch.long,
            device=device,
        )
        dawn_fields = torch.tensor(
            [observation.dawn_tokens for observation in observations],
            dtype=torch.long,
            device=device,
        )

        batch_size = len(observations)
        cls = self.cls_token.expand(batch_size, -1, -1)
        global_token = self.global_encoder(global_fields).unsqueeze(1)
        player_tokens = self.player_encoder(player_fields)
        semantic_tokens = self.semantic_encoder(semantic_fields)
        vote_tokens = self.vote_encoder(vote_fields)
        dawn_tokens = self.dawn_encoder(dawn_fields)
        tokens = torch.cat(
            (cls, global_token, player_tokens, semantic_tokens, vote_tokens, dawn_tokens),
            dim=1,
        )
        positions = torch.arange(_SEQUENCE_LENGTH, device=device).unsqueeze(0)
        tokens = tokens + self.position_embedding(positions)
        padding_mask = self._padding_mask(observations, device=device)
        encoded = self.transformer(tokens, src_key_padding_mask=padding_mask)

        context = encoded[:, 0]
        players = encoded[:, _PLAYER_START:_SEMANTIC_START]
        semantics = encoded[
            :,
            _SEMANTIC_START : _SEMANTIC_START + MAX_SEMANTIC_EVENTS,
        ]
        return TorchPolicyTensorOutput(
            timing=self.timing_head(context),
            action_type=self.action_type_head(context),
            topic=self.topic_head(context),
            target=self.target_pointer(context, players),
            secondary_target=self.secondary_target_pointer(context, players),
            role=self.role_head(context),
            result=self.result_head(context),
            quantity=self.quantity_head(context),
            referenced_day=self.referenced_day_head(context),
            scope=self.scope_head(context),
            stance=self.stance_head(context),
            reference_event=self.reference_event_pointer(context, semantics),
            vote_target=self.vote_target_pointer(context, players),
            night_topic=self.night_topic_head(context),
            night_target=self.night_target_pointer(context, players),
            value=self.value_head(context).squeeze(-1),
        )

    @staticmethod
    def _padding_mask(
        observations: Sequence[EncodedPolicyObservation],
        *,
        device: torch.device,
    ) -> Tensor:
        masks: list[list[bool]] = []
        for observation in observations:
            masks.append(
                [False, False]
                + [False] * MAX_SEATS
                + [not bool(value) for value in observation.semantic_mask]
                + [not bool(value) for value in observation.vote_mask]
                + [not bool(value) for value in observation.dawn_mask]
            )
        return torch.tensor(masks, dtype=torch.bool, device=device)


def _as_tuple(values: Tensor) -> tuple[float, ...]:
    return tuple(float(value) for value in values.detach().cpu().tolist())

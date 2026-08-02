"""Dependency-free, declarative Markdown strategy knowledge loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.engine.roles import RoleName
from app.engine.state import GameState


@dataclass(frozen=True)
class Doctrine:
    metadata: dict[str, str]
    body: str

    @property
    def priority(self) -> int:
        try:
            return int(self.metadata.get("priority", "0"))
        except ValueError:
            return 0


@dataclass(frozen=True)
class KnowledgeContext:
    state: GameState
    player_id: str
    fake_role: RoleName | None = None
    perspective_needed: bool = False


def parse_doctrine(text: str) -> Doctrine:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("knowledge document must start with ---")
    try:
        end = next(i for i, line in enumerate(lines[1:], 1) if line.strip() == "---")
    except StopIteration as exc:
        raise ValueError("knowledge front matter is not closed") from exc
    metadata: dict[str, str] = {}
    for line in lines[1:end]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise ValueError(f"invalid front matter line: {line}")
        key, value = stripped.split(":", 1)
        metadata[key.strip()] = value.split("#", 1)[0].strip()
    if not metadata.get("id"):
        raise ValueError("knowledge document requires id")
    return Doctrine(metadata=metadata, body="\n".join(lines[end + 1 :]).strip())


class KnowledgeBase:
    def __init__(self, directory: Path | None = None) -> None:
        root = directory or Path(__file__).with_name("knowledge")
        self.doctrines = [
            parse_doctrine(path.read_text(encoding="utf-8")) for path in root.glob("*.md")
        ]

    def select(self, context: KnowledgeContext, limit: int = 8) -> list[Doctrine]:
        matched = [item for item in self.doctrines if self._matches(item, context)]
        return sorted(matched, key=lambda item: (-item.priority, item.metadata["id"]))[:limit]

    @staticmethod
    def _matches(doctrine: Doctrine, context: KnowledgeContext) -> bool:
        meta = doctrine.metadata
        state = context.state
        player = state.players[context.player_id]
        if int(meta.get("min_day", "0")) > state.day:
            return False
        roles = _csv(meta.get("player_roles"))
        if roles and player.role.value not in roles:
            return False
        factions = _csv(meta.get("factions"))
        if factions and player.team.value not in factions:
            return False
        if _bool(meta.get("fake_only")) and context.fake_role is None:
            return False
        claimed = _csv(meta.get("claimed_roles"))
        own_claims = {
            claim.claimed_role.value
            for claim in state.co_declarations
            if claim.player_id == context.player_id
        }
        fake_claim = {context.fake_role.value} if context.fake_role else set()
        if claimed and not claimed.intersection(own_claims | fake_claim):
            return False
        if condition := meta.get("min_co_count"):
            role_name, threshold = condition.split(">=", 1)
            count = sum(c.claimed_role.value == role_name.strip() for c in state.co_declarations)
            if count < int(threshold):
                return False
        if _bool(meta.get("perspective_only")) and not context.perspective_needed:
            return False
        return True


def _csv(value: str | None) -> set[str]:
    return {part.strip() for part in (value or "").split(",") if part.strip()}


def _bool(value: str | None) -> bool:
    return (value or "").lower() == "true"

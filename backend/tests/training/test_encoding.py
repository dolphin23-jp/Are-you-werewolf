from app.engine.game import PlayerSpec
from app.engine.roles import RoleName
from app.training.actions import ActionType, SemanticAction, SpeechBundle, Topic
from app.training.encoding import (
    MAX_DAWN_EVENTS,
    MAX_SEATS,
    MAX_SEMANTIC_EVENTS,
    MAX_VOTE_EVENTS,
    ObservationEncoder,
)
from app.training.env import WerewolfTrainingEnv


def _env() -> WerewolfTrainingEnv:
    specs = [PlayerSpec(player_id=f"p{i}", name=f"Player{i}") for i in range(17)]
    return WerewolfTrainingEnv(
        specs,
        seed=23,
        forced_roles={
            "p0": RoleName.SEER,
            "p1": RoleName.WEREWOLF,
            "p2": RoleName.FOX,
        },
    )


def test_encoder_has_fixed_shapes_and_explicit_padding_masks():
    encoded = ObservationEncoder().encode(_env().observe("p0"))

    assert len(encoded.player_tokens) == MAX_SEATS
    assert len(encoded.semantic_tokens) == MAX_SEMANTIC_EVENTS
    assert len(encoded.semantic_mask) == MAX_SEMANTIC_EVENTS
    assert len(encoded.vote_tokens) == MAX_VOTE_EVENTS
    assert len(encoded.vote_mask) == MAX_VOTE_EVENTS
    assert len(encoded.dawn_tokens) == MAX_DAWN_EVENTS
    assert len(encoded.dawn_mask) == MAX_DAWN_EVENTS
    assert sum(encoded.semantic_mask) == 0
    assert sum(encoded.vote_mask) == 0


def test_encoder_includes_public_semantics_but_not_true_roles_of_other_seats():
    env = _env()
    env.controller.resolve_night()
    env.controller.start_discussion()
    env.emit_speech(
        "p1",
        SpeechBundle(
            (
                SemanticAction(
                    ActionType.CLAIM,
                    topic=Topic.ROLE,
                    role=RoleName.SEER,
                ),
            )
        ),
    )

    encoded = ObservationEncoder().encode(env.observe("p2"))

    assert sum(encoded.semantic_mask) == 1
    # Player token field 5 is the public current CO, not the true role. p1 is
    # truly a werewolf but publicly claims seer here.
    p1 = encoded.player_tokens[1]
    assert p1[6] == 0  # fox does not know p1 as an ally
    assert p1[5] != 0  # a public role claim is encoded


def test_private_result_is_encoded_only_for_the_entitled_viewer():
    env = _env()
    env.controller.submit_night_action("p0", "divine", "p1")
    env.controller.resolve_night()

    seer = ObservationEncoder().encode(env.observe("p0"))
    fox = ObservationEncoder().encode(env.observe("p2"))

    p1_seer = seer.player_tokens[1]
    p1_fox = fox.player_tokens[1]
    assert p1_seer[7] == 2
    assert p1_fox[7] == 0

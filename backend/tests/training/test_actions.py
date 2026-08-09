import pytest

from app.training.actions import ActionType, SemanticAction, SpeechBundle


def test_speech_bundle_accepts_one_to_three_semantic_atoms():
    SpeechBundle((SemanticAction(ActionType.CLAIM),))
    SpeechBundle(
        (
            SemanticAction(ActionType.CLAIM),
            SemanticAction(ActionType.REPORT),
            SemanticAction(ActionType.EVALUATE),
        )
    )


@pytest.mark.parametrize("action_type", [ActionType.VOTE, ActionType.NIGHT_ACTION])
def test_speech_bundle_rejects_execution_actions(action_type):
    with pytest.raises(ValueError):
        SpeechBundle((SemanticAction(action_type),))

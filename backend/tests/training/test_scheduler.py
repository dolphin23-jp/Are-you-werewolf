from app.training.actions import ActionType, SemanticAction, SpeechBundle, TimingBucket
from app.training.scheduler import EventDrivenDiscussionScheduler, SpeakIntent


def _speech(action_type: ActionType = ActionType.CLAIM) -> SpeechBundle:
    return SpeechBundle((SemanticAction(action_type),))


def test_scheduler_selects_earliest_logical_timing_and_ignores_hold():
    scheduler = EventDrivenDiscussionScheduler(seed=1)
    selected = scheduler.select_next(
        {
            "p1": SpeakIntent(TimingBucket.LATE, _speech()),
            "p2": SpeakIntent(TimingBucket.IMMEDIATE, _speech()),
            "p3": SpeakIntent(TimingBucket.HOLD, _speech()),
        }
    )

    assert selected is not None
    assert selected.player_id == "p2"
    assert selected.timing is TimingBucket.IMMEDIATE
    assert selected.discussion_tick == 0


def test_scheduler_does_not_preserve_old_intents_across_events():
    scheduler = EventDrivenDiscussionScheduler(seed=1)
    first = scheduler.select_next(
        {
            "wolf": SpeakIntent(TimingBucket.NORMAL, _speech()),
            "seer": SpeakIntent(TimingBucket.IMMEDIATE, _speech()),
        }
    )
    assert first is not None and first.player_id == "seer"

    scheduler.record_emitted_event()
    # Caller replans from the new public state. The wolf can now abandon the
    # claim it had intended to make before seeing the seer's CO.
    second = scheduler.select_next(
        {
            "wolf": SpeakIntent(TimingBucket.HOLD, None),
            "medium": SpeakIntent(TimingBucket.EARLY, _speech()),
        }
    )
    assert second is not None and second.player_id == "medium"
    assert second.discussion_tick == 1

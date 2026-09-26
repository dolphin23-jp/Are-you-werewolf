import argparse
import io

from app.training.resume_flags import warn_ignored_on_resume


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=20)
    return parser


def test_resume_names_explicit_flags_it_will_not_apply():
    stream = io.StringIO()
    argv = ["--resume", "--learning-rate=1e-4", "--ppo-epochs", "4", "--episodes", "40"]

    ignored = warn_ignored_on_resume(
        _parser(), argv, ("learning_rate", "ppo_epochs"), stream=stream
    )

    assert ignored == ("--learning-rate", "--ppo-epochs")
    assert "--learning-rate, --ppo-epochs" in stream.getvalue()


def test_resume_is_quiet_when_only_applied_flags_are_passed():
    stream = io.StringIO()

    ignored = warn_ignored_on_resume(
        _parser(), ["--resume", "--episodes", "40"], ("learning_rate",), stream=stream
    )

    assert ignored == ()
    assert stream.getvalue() == ""

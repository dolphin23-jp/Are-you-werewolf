"""Every metrics event the app can record has a label in the report.

The report's code-substitution table lists every `EVENT_LABELS` entry, zero
included, and shows anything else as a bare backticked name. A new event added
without a label therefore never fails a run; it just turns up unexplained in
the report, as `designation_question_unanswered` did. This reads the call sites
instead of a hand-kept list, so adding an event and forgetting its label fails
here.
"""

from __future__ import annotations

import ast
from pathlib import Path

from app.eval.report import EVENT_LABELS

APP_ROOT = Path(__file__).resolve().parents[2] / "app"
_RECORDERS = frozenset({"record_event", "_record_event"})


def _recorded_event_names() -> dict[str, list[str]]:
    """Event name -> the call sites that record it, read from the app's source.

    A call whose name is not a string literal is reported as a site of its own
    (`<non-literal>`), so an event name built at runtime cannot slip past.
    """
    names: dict[str, list[str]] = {}
    for path in sorted(APP_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if called not in _RECORDERS:
                continue
            argument = node.args[0]
            site = f"{path.relative_to(APP_ROOT.parent)}:{node.lineno}"
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                names.setdefault(argument.value, []).append(site)
            else:
                names.setdefault("<non-literal>", []).append(site)
    return names


def test_the_scan_finds_the_known_call_sites():
    # Guards the scan itself: if it found nothing, the next test would pass
    # vacuously.
    recorded = _recorded_event_names()

    assert "vote_code_resolved" in recorded
    assert "discussion_stalled" in recorded
    assert "designation_question_unanswered" in recorded


def test_every_recorded_event_has_a_report_label():
    recorded = _recorded_event_names()

    assert "<non-literal>" not in recorded, recorded.get("<non-literal>")
    unlabelled = {name: sites for name, sites in recorded.items() if name not in EVENT_LABELS}
    assert unlabelled == {}


def test_every_label_names_an_event_the_app_records():
    # A label for an event nothing records is a row that always reads 0 and
    # looks like a measurement.
    recorded = _recorded_event_names()

    assert set(EVENT_LABELS) - set(recorded) == set()


def test_an_unanswerable_designation_question_reads_in_words():
    assert EVENT_LABELS["designation_question_unanswered"] == "本指定で出た、答える番の無い質問"

"""Which optional parameters an OpenAI-compatible endpoint actually accepts.

The "OpenAI-compatible" label hides real incompatibilities between model
generations, and they surface as hard 400s rather than being ignored:

    max_tokens        -> newer models reject it, demanding
                         max_completion_tokens instead
    temperature=0.9   -> newer models reject any non-default value

Older or third-party endpoints want exactly the opposite. Making the
operator configure this by hand means a failed run to discover each one,
so instead we start with the modern spelling and learn from the rejection:
`EndpointDialect.adapt()` reads the offending parameter out of the error
and flips that single setting, leaving everything else alone.

The learned dialect is remembered on the provider instance, so one
rejected call per process pays for the whole run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Codes an OpenAI-shaped API uses to say "that parameter is not for you".
_REJECTION_CODES = {"unsupported_parameter", "unsupported_value"}

_TOKEN_PARAMS = ("max_completion_tokens", "max_tokens")

_PARAM_IN_MESSAGE_RE = re.compile(
    r"(?:Unsupported parameter|Unsupported value|Unrecognized request argument)"
    r"[^']*'(?P<param>[a-z_]+)'",
    re.IGNORECASE,
)


def rejected_parameter(exc: Exception) -> str | None:
    """The parameter name an API error is complaining about, if any."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            param = error.get("param")
            code = error.get("code")
            if param and (code in _REJECTION_CODES or code is None):
                return str(param)

    # Some gateways forward only the message text.
    match = _PARAM_IN_MESSAGE_RE.search(str(exc))
    return match.group("param") if match else None


@dataclass
class EndpointDialect:
    """Mutable per-provider record of what this endpoint tolerates."""

    token_param: str = "max_completion_tokens"
    send_temperature: bool = True

    def apply(self, kwargs: dict[str, Any], *, max_tokens: int, temperature: float) -> None:
        kwargs[self.token_param] = max_tokens
        if self.send_temperature:
            kwargs["temperature"] = temperature
        else:
            kwargs.pop("temperature", None)

    def adapt(self, exc: Exception) -> bool:
        """Adjust to an API rejection. Returns True when something changed,
        meaning the same call is worth retrying once."""
        param = rejected_parameter(exc)
        if param is None:
            return False

        if param in _TOKEN_PARAMS:
            other = _TOKEN_PARAMS[1] if param == _TOKEN_PARAMS[0] else _TOKEN_PARAMS[0]
            if self.token_param == other:
                return False  # already using the other spelling; nothing left to try
            self.token_param = other
            return True

        if param == "temperature" and self.send_temperature:
            self.send_temperature = False
            return True

        return False

    def describe(self) -> str:
        temp = "送信する" if self.send_temperature else "送信しない(既定値のみ対応)"
        return f"トークン上限パラメータ={self.token_param} / temperature={temp}"

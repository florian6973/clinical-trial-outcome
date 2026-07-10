"""Strict parser for the Qwen ``GROUP:`` output contract."""

from __future__ import annotations

import re


class GroupParseError(ValueError):
    """Raised when Qwen does not return one valid group line."""


_GROUP_LINE = re.compile(r"^\s*GROUP\s*:\s*(.*?)\s*$", flags=re.IGNORECASE)
_SPECIAL_TOKENS = ("</s>", "<|endoftext|>", "<|im_end|>", "<|eot_id|>")


def parse_group_response(response: str) -> str:
    """Return the first and only populated ``GROUP:`` value."""

    matches: list[str] = []
    for line in response.splitlines():
        match = _GROUP_LINE.match(line)
        if match:
            group = match.group(1)
            for token in _SPECIAL_TOKENS:
                group = group.replace(token, "")
            group = " ".join(group.split()).strip("`'\"")
            if group:
                matches.append(group)
    if not matches:
        raise GroupParseError("Response must contain a populated 'GROUP:' line")
    if len(matches) > 1:
        raise GroupParseError("Response contains more than one 'GROUP:' line")
    if matches[0].casefold() in {"[group name]", "group name", "none", "n/a"}:
        raise GroupParseError("Response contains a placeholder instead of a group")
    return matches[0]

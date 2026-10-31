"""Colour tokens for charts and tiles, selected per light/dark mode (not flipped).

Measures keep one colour everywhere: pace is blue, HR orange, everything else (mechanics,
efficiency) aqua. Ordered series (rep 1..n, block 1..n) use a single-hue blue ramp that
gets stronger with each rep, so later reps stand out.

The three series colours and the ordinal ramp come from the dataviz skill's reference
palette and were run through its validator in both modes. Aqua is below 3:1 on the light
surface, so every chart that uses it has a table or labels next to it.
"""

from dataclasses import dataclass

import numpy as np
import panel as pn


@dataclass(frozen=True)
class Palette:
    surface: str
    ink: str
    ink2: str
    muted: str
    grid: str
    border: str
    pace: str
    hr: str
    other: str
    ordinal: tuple[str, ...]  # first rep -> last rep


LIGHT = Palette(
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink2="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    border="rgba(11,11,11,0.10)",
    pace="#2a78d6",
    hr="#eb6834",
    other="#1baf7a",
    # blue steps 250 -> 700
    ordinal=(
        "#86b6ef",
        "#6da7ec",
        "#5598e7",
        "#3987e5",
        "#2a78d6",
        "#256abf",
        "#1c5cab",
        "#184f95",
        "#104281",
        "#0d366b",
    ),
)

DARK = Palette(
    surface="#1a1a19",
    ink="#f0efec",
    ink2="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    border="rgba(255,255,255,0.10)",
    pace="#3987e5",
    hr="#d95926",
    other="#199e70",
    # blue steps 600 -> 100: on a dark surface stronger means lighter
    ordinal=(
        "#184f95",
        "#1c5cab",
        "#256abf",
        "#2a78d6",
        "#3987e5",
        "#5598e7",
        "#6da7ec",
        "#86b6ef",
        "#9ec5f4",
        "#b7d3f6",
    ),
)

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def is_dark() -> bool:
    """The template's theme toggle reloads the page with ?theme=dark."""
    args = pn.state.session_args or {}
    value = (args.get("theme") or [b""])[0]
    if isinstance(value, bytes):
        value = value.decode()
    return value == "dark" or pn.config.theme == "dark"


def palette() -> Palette:
    return DARK if is_dark() else LIGHT


MAX_ORDINAL = 5  # beyond this, ramp steps are too close in lightness to tell apart


def ordinal(n: int) -> list[str]:
    """n colours in order. Up to MAX_ORDINAL they are spread along the ramp (at least two steps
    apart, which passes the palette validator); with more, only the first and last get the
    ramp's ends and the rest are muted, keeping the first-vs-last comparison readable."""
    p = palette()
    ramp = p.ordinal
    if n <= 1:
        return [ramp[len(ramp) // 2]]
    if n <= MAX_ORDINAL:
        return [ramp[int(i)] for i in np.linspace(0, len(ramp) - 1, n).round()]
    return [ramp[0], *[p.muted] * (n - 2), ramp[-1]]


def stylesheet() -> str:
    """CSS for the HTML bits (stat tiles, chips), using the palette as custom properties."""
    p = palette()
    return f"""
    :host {{
      --surface: {p.surface}; --ink: {p.ink}; --ink2: {p.ink2}; --muted: {p.muted};
      --border: {p.border}; --accent: {p.pace};
      font-family: {FONT}; color: var(--ink);
    }}
    .tiles {{ display: flex; flex-wrap: wrap; gap: 12px; }}
    .tile {{ flex: 1 1 140px; padding: 10px 14px; border: 1px solid var(--border);
             border-radius: 8px; background: var(--surface); }}
    .tile .label {{ font-size: 12px; color: var(--ink2); }}
    .tile .value {{ font-size: 24px; font-weight: 600; line-height: 1.3; }}
    .tile .unit {{ font-size: 13px; font-weight: 400; color: var(--ink2); margin-left: 2px; }}
    .tile .note {{ font-size: 12px; color: var(--muted); }}
    .chip {{ display: inline-block; padding: 1px 8px; margin-right: 4px; border-radius: 10px;
             border: 1px solid var(--border); font-size: 12px; color: var(--ink2); }}
    .meta {{ color: var(--ink2); font-size: 14px; }}
    .meta b {{ color: var(--ink); font-weight: 600; }}
    .hint {{ color: var(--muted); font-size: 13px; }}
    """

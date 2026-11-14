"""Startup banner for the DPDFNet CLI."""
from __future__ import annotations

import os
import sys

_ASCII = r"""
██████╗ ██████╗ ██████╗ ███████╗███╗   ██╗███████╗████████╗
██╔══██╗██╔══██╗██╔══██╗██╔════╝████╗  ██║██╔════╝╚══██╔══╝
██║  ██║██████╔╝██║  ██║█████╗  ██╔██╗ ██║█████╗     ██║
██║  ██║██╔═══╝ ██║  ██║██╔══╝  ██║╚██╗██║██╔══╝     ██║
██████╔╝██║     ██████╔╝██║     ██║ ╚████║███████╗   ██║
╚═════╝ ╚═╝     ╚═════╝ ╚═╝     ╚═╝  ╚═══╝╚══════╝   ╚═╝
"""


def _ansi(code: str) -> str:
    return f"\x1b[{code}m"


RESET = _ansi("0")
BOLD = _ansi("1")
DIM = _ansi("2")
LIGHT_GRAY = _ansi("90")


def print_banner(
    *,
    model_name: str,
    sample_rate: int,
    description: str,
) -> None:
    """Print the DPDFNet ASCII banner followed by model info lines.

    Respects:
      - NO_BANNER=1/true/yes  → skip entirely
      - non-TTY stderr        → skip entirely
    """
    if os.getenv("NO_BANNER", "").lower() in {"1", "true", "yes"}:
        return
    if not sys.stderr.isatty():
        return

    title = "DPDFNet"
    tagline = "Powered by CEVA Inc."

    content_lines = [ln.rstrip("\n") for ln in _ASCII.strip("\n").splitlines()]
    content_lines += ["", title, tagline]

    inner_w = max(len(ln) for ln in content_lines)

    tl, tr, bl, br = "┌", "┐", "└", "┘"
    h, v = "─", "│"

    top = f"{tl}{h * (inner_w + 2)}{tr}"
    bot = f"{bl}{h * (inner_w + 2)}{br}"

    framed = [top]
    for ln in content_lines:
        framed.append(f"{v} {ln.ljust(inner_w)} {v}")
    framed.append(bot)

    # Style the framed banner
    styled: list[str] = []
    for i, ln in enumerate(framed):
        if i == 0 or i == len(framed) - 1:
            styled.append(f"{LIGHT_GRAY}{ln}{RESET}")
            continue

        ci = i - 1
        raw = content_lines[ci]

        if raw == title:
            styled_ln = ln.replace(
                raw, f"{BOLD}{LIGHT_GRAY}{raw}{RESET}{LIGHT_GRAY}", 1
            )
            styled.append(f"{LIGHT_GRAY}{styled_ln}{RESET}")
        elif raw == tagline:
            styled_ln = ln.replace(
                raw, f"{DIM}{LIGHT_GRAY}{raw}{RESET}{LIGHT_GRAY}", 1
            )
            styled.append(f"{LIGHT_GRAY}{styled_ln}{RESET}")
        else:
            styled.append(f"{LIGHT_GRAY}{ln}{RESET}")

    # Info lines outside the frame
    sr_khz = f"{sample_rate // 1000} kHz"
    info_lines = [
        f"  {BOLD}Model:{RESET}       {model_name}",
        f"  {BOLD}Sample rate:{RESET} {sr_khz}",
        f"  {BOLD}Description:{RESET} {description}",
    ]

    output = "\n".join(styled) + "\n" + "\n".join(info_lines) + "\n"
    print(output, file=sys.stderr, flush=True)

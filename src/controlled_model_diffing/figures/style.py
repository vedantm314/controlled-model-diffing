"""Shared figure setup.

Every figure renders headless and writes a PNG into results/.

Patchscope surfaces CJK and Turkish tokens, which DejaVu cannot draw. The Noto
CJK font is not in git because of its size. Fetch it with:

  curl -fsSL https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/Japanese/\\
NotoSansCJKjp-Regular.otf -o assets/NotoSansCJK.otf

Without it the figures still render, and the CJK tokens become boxes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from controlled_model_diffing.paths import ASSETS_DIR, RESULTS_DIR  # noqa: E402

CJK_FONT = ASSETS_DIR / "NotoSansCJK.otf"


def use_cjk_font() -> bool:
    """Register the CJK font if it is present. Returns whether it was."""
    import matplotlib.font_manager as fm

    if not CJK_FONT.exists():
        return False
    fm.fontManager.addfont(str(CJK_FONT))
    matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Noto Sans CJK JP"]
    return True


def save(fig, name: str, results_dir: Path | None = None, dpi: int = 200) -> Path:
    d = results_dir or RESULTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / name
    fig.savefig(path, dpi=dpi)
    print(f"wrote {path}")
    return path


def token_column(ax, tokens: list[str], title: str, highlight, n: int = 21) -> None:
    """One ranked list of tokens as a text column, relevant tokens coloured."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(title)
    for i, tk in enumerate(tokens):
        ax.text(0.05, i + 1, f"{i+1:2d}  {tk}", ha="left", va="center", fontsize=9,
                color="tab:orange" if highlight.search(tk) else "black")

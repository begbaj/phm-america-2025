# predictor - Modular PHM prediction pipeline

import os
from pathlib import Path

import matplotlib.pyplot as plt

from predictor import config as cfg

_plot_counter: dict[str, int] = {}


def save_fig(fig: plt.Figure, name: str, *, dpi: int = 150) -> None:
    """Save *fig* to ``cfg.PLOTS_DIR/<name>.png`` and close it.

    If a file with the same *name* already exists, a numeric suffix is
    appended automatically (e.g. ``name_2.png``, ``name_3.png``).
    """
    Path(cfg.PLOTS_DIR).mkdir(parents=True, exist_ok=True)

    # Auto-increment duplicates
    if name in _plot_counter:
        _plot_counter[name] += 1
        fname = f"{name}_{_plot_counter[name]}.png"
    else:
        _plot_counter[name] = 1
        fname = f"{name}.png"

    path = os.path.join(cfg.PLOTS_DIR, fname)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

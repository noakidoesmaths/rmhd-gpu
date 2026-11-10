"""Small helpers for consistent save/show plotting behavior.

The `vis/` scripts save figures to disk by default. When `--show` is used, the
helper leaves the interactive backend alone when possible so the same script
works naturally from Spyder, IPython, or a normal command line session.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib


_INTERACTIVE_BACKENDS = ("QtAgg", "TkAgg", "MacOSX")


def import_pyplot(*, show: bool) -> Any:
    """Import pyplot after choosing an appropriate backend.

    Without `--show`, the scripts force the noninteractive `Agg` backend so
    they behave as batch plot writers. With `--show`, they keep the active
    backend. If pyplot is already on `Agg`, the helper makes a best-effort
    switch to a common interactive backend so repeated Spyder runs can still
    display figures.
    """

    if not show:
        matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt

    if show and "agg" in matplotlib.get_backend().lower():
        for backend_name in _INTERACTIVE_BACKENDS:
            try:
                plt.switch_backend(backend_name)
                break
            except Exception:
                continue

    return plt


def finalize_figure(fig: Any, *, output_path: Path, show: bool, plt: Any) -> None:
    """Save one figure, optionally show it, and then close it."""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    print(f"Saved {output_path}")
    if show:
        plt.show()
    plt.close(fig)

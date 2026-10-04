"""Layout and colour-scale checks for vis/movie_fullfield.py."""

from __future__ import annotations

import numpy as np
import pytest

from vis import movie_fullfield as mf

h5py = pytest.importorskip("h5py")


def _write_snapshot(path, *, field: np.ndarray, time: float, step: int) -> None:
    n = field.shape[0]
    coords = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    with h5py.File(path, "w") as handle:
        for name in ("x", "y", "z"):
            handle.create_dataset(f"metadata/{name}", data=coords)
        handle.create_dataset("output/psi", data=field)
        handle.create_dataset("output/time", data=time)
        handle.create_dataset("output/step", data=step)


def test_freeze_layout_keeps_axes_and_colorbar_fixed_when_title_changes():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    image = ax.imshow(np.random.default_rng(0).standard_normal((8, 8)), vmin=-1, vmax=1, aspect="auto")
    colorbar = fig.colorbar(image, ax=ax)
    title = ax.set_title("")

    texts = [f"psi  z=0.000 (index 0)  t={t:.3f}  step={s}" for t, s in [(0.0, 0), (1000.123, 100000)]]
    mf._freeze_layout(fig, title, texts)
    ax_before = np.array(ax.get_position().bounds)
    cbar_before = np.array(colorbar.ax.get_position().bounds)

    for text in texts + ["x" * 5, "a much wider title " * 3]:
        title.set_text(text)
        fig.canvas.draw()
        np.testing.assert_array_equal(ax.get_position().bounds, ax_before)
        np.testing.assert_array_equal(colorbar.ax.get_position().bounds, cbar_before)
    plt.close(fig)


def test_script_writes_gif_with_global_color_scale(tmp_path):
    n = 8
    rng = np.random.default_rng(1)
    for index, (amp, time, step) in enumerate([(1.0, 0.0, 0), (3.0, 10.0, 1000)], start=1):
        _write_snapshot(
            tmp_path / f"fullfield_{index:04d}.h5",
            field=amp * rng.standard_normal((n, n, n)),
            time=time,
            step=step,
        )
    frames, *_ = mf._load_frames(
        sorted(tmp_path.glob("fullfield_*.h5")),
        field="psi",
        slice_dir="z",
        slice_index_arg=None,
        slice_coord_arg=None,
    )
    assert len(frames) == 2

    output = tmp_path / "movie.gif"
    result = mf.main([str(tmp_path), "--field", "psi", "--output", str(output), "--dpi", "40"])
    assert result == output.resolve()
    assert output.exists() and output.stat().st_size > 0

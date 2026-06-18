"""Animate 2D slices from per-snapshot full-field HDF5 outputs into a movie.

This reads the `fullfield_*.h5` snapshots written by `FullFieldHDF5Writer`
(see `rmhdgpu/output.py`) and stitches a chosen 2D slice across all snapshots
into a single movie file, ordered by snapshot number.

Examples
--------
    python vis/movie_fullfield.py examples/outputs/fullfields --field psi
    python vis/movie_fullfield.py examples/outputs/fullfields \
        --field omega --slice-dir z --fps 15 --output omega.mp4

The output format is chosen from the `--output` extension: `.mp4` (requires
ffmpeg on PATH) or `.gif` (uses matplotlib's bundled Pillow writer). If no
`--output` is given, an `.mp4` is written next to the snapshots and the code
falls back to `.gif` when ffmpeg is unavailable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from vis._matplotlib import import_pyplot

try:
    import h5py
except ImportError:  # pragma: no cover - exercised by runtime error path
    h5py = None


def _require_h5py() -> None:
    if h5py is None:
        raise SystemExit("movie_fullfield.py requires `h5py` to read full-field HDF5 snapshots.")


def _resolve_input_files(path: Path) -> list[Path]:
    if path.is_dir():
        files = sorted(path.glob("fullfield_*.h5"))
        if not files:
            raise SystemExit(f"No full-field snapshot files were found in {path}.")
        return files
    if path.suffix != ".h5":
        raise SystemExit(f"Expected a snapshot .h5 file or a directory of snapshots; got {path}.")
    return [path]


def _resolve_slice_index(coords: np.ndarray, *, requested_index: int | None, requested_coord: float | None) -> int:
    if requested_coord is not None:
        return int(np.argmin(np.abs(coords - requested_coord)))
    if requested_index is not None:
        if requested_index < 0 or requested_index >= len(coords):
            raise SystemExit(f"slice index {requested_index} is out of range for axis of length {len(coords)}.")
        return requested_index
    return len(coords) // 2


def _extract_slice(field: np.ndarray, *, slice_dir: str, slice_index: int) -> np.ndarray:
    if slice_dir == "x":
        return field[slice_index, :, :]
    if slice_dir == "y":
        return field[:, slice_index, :]
    return field[:, :, slice_index]


def _slice_axes(
    *,
    slice_dir: str,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[str, str, tuple[float, float, float, float]]:
    if slice_dir == "x":
        return "y", "z", (float(y[0]), float(y[-1]), float(z[0]), float(z[-1]))
    if slice_dir == "y":
        return "x", "z", (float(x[0]), float(x[-1]), float(z[0]), float(z[-1]))
    return "x", "y", (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "input_path",
        help="Path to a directory of `fullfield_*.h5` files (or a single `.h5` file).",
    )
    parser.add_argument("--field", default="psi", help="Field to animate (e.g. psi, omega, du_par, db_par, s).")
    parser.add_argument("--slice-dir", choices=["x", "y", "z"], default="z", help="Slice direction.")
    parser.add_argument("--slice-index", type=int, default=None, help="Explicit slice index.")
    parser.add_argument(
        "--slice-coordinate",
        type=float,
        default=None,
        help="Slice coordinate. The nearest stored plane is selected.",
    )
    parser.add_argument(
        "--indices",
        nargs="*",
        type=int,
        default=None,
        help="Optional subset of snapshot numbers to include, matching file names such as `1`.",
    )
    parser.add_argument("--output", default=None, help="Output movie path (.mp4 or .gif).")
    parser.add_argument("--fps", type=int, default=12, help="Frames per second.")
    parser.add_argument("--cmap", default="RdBu_r", help="Matplotlib colormap name.")
    parser.add_argument("--dpi", type=int, default=140, help="Output resolution in dots per inch.")
    parser.add_argument(
        "--symmetric",
        action="store_true",
        default=True,
        help="Use a symmetric color scale centered on zero (default for signed fields).",
    )
    parser.add_argument(
        "--no-symmetric",
        dest="symmetric",
        action="store_false",
        help="Use the raw [min, max] data range instead of a symmetric scale.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the animation interactively instead of (or in addition to) saving.",
    )
    return parser


def _load_frames(
    snapshot_files: list[Path],
    *,
    field: str,
    slice_dir: str,
    slice_index_arg: int | None,
    slice_coord_arg: float | None,
) -> tuple[list[tuple[str, float, int, np.ndarray]], np.ndarray, np.ndarray, np.ndarray, int]:
    frames: list[tuple[str, float, int, np.ndarray]] = []
    x = y = z = None
    slice_index = 0
    for snapshot_path in snapshot_files:
        with h5py.File(snapshot_path, "r") as handle:
            metadata = handle["metadata"]
            output_group = handle["output"]

            if x is None:
                x = np.asarray(metadata["x"])
                y = np.asarray(metadata["y"])
                z = np.asarray(metadata["z"])
                slice_index = _resolve_slice_index(
                    {"x": x, "y": y, "z": z}[slice_dir],
                    requested_index=slice_index_arg,
                    requested_coord=slice_coord_arg,
                )
            if field not in output_group:
                available = ", ".join(k for k in output_group if k not in {"time", "step"})
                raise SystemExit(f"Field {field!r} is not present in {snapshot_path}. Available: {available}.")

            field_data = np.asarray(output_group[field])
            slice_data = _extract_slice(field_data, slice_dir=slice_dir, slice_index=slice_index)
            time_value = float(np.asarray(output_group["time"]))
            step_value = int(np.asarray(output_group["step"]))
            frames.append((snapshot_path.stem.split("_")[-1], time_value, step_value, slice_data))

    assert x is not None and y is not None and z is not None
    return frames, x, y, z, slice_index


def main(argv: list[str] | None = None) -> Path:
    _require_h5py()
    args = build_parser().parse_args(argv)
    plt = import_pyplot(show=args.show)
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

    input_path = Path(args.input_path).expanduser().resolve()
    snapshot_files = _resolve_input_files(input_path)
    if args.indices is not None:
        requested_names = {f"fullfield_{index:04d}.h5" for index in args.indices}
        snapshot_files = [path for path in snapshot_files if path.name in requested_names]
        if not snapshot_files:
            raise SystemExit(f"Requested snapshot numbers were not present in {input_path}.")

    frames, x, y, z, slice_index = _load_frames(
        snapshot_files,
        field=args.field,
        slice_dir=args.slice_dir,
        slice_index_arg=args.slice_index,
        slice_coord_arg=args.slice_coordinate,
    )

    # Fix the color scale across all frames so the animation is comparable
    # frame-to-frame rather than re-normalizing each one.
    if args.symmetric:
        vmax = max((float(np.max(np.abs(data))) for _, _, _, data in frames), default=0.0)
        vmax = 1.0 if vmax == 0.0 else vmax
        vmin = -vmax
    else:
        vmin = min((float(np.min(data)) for _, _, _, data in frames), default=0.0)
        vmax = max((float(np.max(data)) for _, _, _, data in frames), default=1.0)
        if vmin == vmax:
            vmin, vmax = vmin - 1.0, vmax + 1.0

    xlabel, ylabel, extent = _slice_axes(slice_dir=args.slice_dir, x=x, y=y, z=z)

    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    first = frames[0][3]
    image = ax.imshow(
        first.T,
        origin="lower",
        cmap=args.cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        aspect="auto",
    )
    fig.colorbar(image, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    title = ax.set_title("")

    def update(frame_index: int):
        _key, time_value, step_value, slice_data = frames[frame_index]
        image.set_data(slice_data.T)
        title.set_text(
            f"{args.field}  {args.slice_dir}={slice_index}  "
            f"t={time_value:.3f}  step={step_value}"
        )
        return image, title

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000.0 / args.fps, blit=False)

    if args.output is None:
        base_dir = input_path if input_path.is_dir() else input_path.parent
        output_path = base_dir / f"{args.field}_{args.slice_dir}_movie.mp4"
    else:
        output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer: object
    if output_path.suffix.lower() == ".gif":
        writer = PillowWriter(fps=args.fps)
    elif FFMpegWriter.isAvailable():
        writer = FFMpegWriter(fps=args.fps, bitrate=-1)
    else:
        output_path = output_path.with_suffix(".gif")
        print("ffmpeg not found on PATH; falling back to an animated GIF.")
        writer = PillowWriter(fps=args.fps)

    anim.save(str(output_path), writer=writer, dpi=args.dpi)
    print(f"Saved {output_path} ({len(frames)} frames)")

    if args.show:
        plt.show()
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    main()

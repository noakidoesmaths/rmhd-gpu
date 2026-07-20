"""Persistent diagnostics output helpers.

Scalar diagnostics are written to CSV, spectra to tidy/long CSV, and
full-field snapshots to one HDF5 file per output event when `h5py`
is available.

Each full-field HDF5 snapshot has the layout:

- `/metadata` group with grid and box metadata stored as attributes
- `/metadata/x`, `/metadata/y`, `/metadata/z` coordinate arrays
- `/metadata/field_names` UTF-8 string dataset
- `/output/time`
- `/output/step`
- `/output/<field_name>` for each saved real-space field
"""

from __future__ import annotations

import csv
import importlib
from pathlib import Path
from typing import Any

import numpy as np

from rmhdgpu.diagnostics.fullfield import extract_full_fields


SCALAR_DIAGNOSTICS_FILENAME = "scalar_diagnostics.csv"
SPECTRA_DIAGNOSTICS_FILENAME = "spectra.csv"
SPECTRA_PRL_DIAGNOSTICS_FILENAME = "spectra_prl.csv"
FULLFIELD_DIAGNOSTICS_DIRNAME = "fullfields"
OUTPUT_TIME_TOLERANCE = 1.0e-15


def output_cadence_enabled(cadence: float | None) -> bool:
    """Return `True` when an output cadence requests persistent output."""

    return cadence is not None and cadence > 0.0


def initial_output_time(cadence: float | None) -> float | None:
    """Return the first output time for a cadence, including the initial state."""

    if not output_cadence_enabled(cadence):
        return None
    return 0.0


def output_due(
    *,
    time: float,
    next_output_time: float | None,
    tmax: float,
) -> bool:
    """Return `True` when an output should be written at the current state."""

    if next_output_time is None:
        return False
    return time >= next_output_time - OUTPUT_TIME_TOLERANCE or time >= tmax - OUTPUT_TIME_TOLERANCE


def advance_output_time(
    *,
    next_output_time: float | None,
    cadence: float | None,
    current_time: float,
) -> float | None:
    """Advance a cadence target beyond the current simulation time."""

    if not output_cadence_enabled(cadence):
        return None

    candidate = float(0.0 if next_output_time is None else next_output_time) + float(cadence)
    while candidate <= current_time + OUTPUT_TIME_TOLERANCE:
        candidate += float(cadence)
    return candidate


class ScalarDiagnosticsWriter:
    """Append scalar diagnostics rows to a CSV file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._handle = self.path.open("w", encoding="utf-8", newline="")
        self._writer: csv.DictWriter[str] | None = None

    def write_row(self, row: dict[str, float | int]) -> None:
        if self._writer is None:
            self._writer = csv.DictWriter(self._handle, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class SpectraDiagnosticsWriter:
    """Append tidy/long shell spectra rows to a CSV file.

    `k_column` names the wavenumber column ("kperp" for perpendicular
    spectra, "kprl" for parallel spectra) and must match the key holding the
    shell centers in the spectra dict passed to `write_spectra`.
    """

    def __init__(self, path: str | Path, k_column: str = "kperp") -> None:
        self.path = Path(path)
        self.k_column = k_column
        self._handle = self.path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(
            self._handle,
            fieldnames=["time", "step", "quantity", k_column, "value"],
        )
        self._writer.writeheader()

    def write_spectra(
        self,
        *,
        time: float,
        step: int,
        spectra: dict[str, np.ndarray],
    ) -> None:
        k_values = np.asarray(spectra[self.k_column], dtype=np.float64)
        for quantity in [key for key in spectra if key != self.k_column]:
            values = np.asarray(spectra[quantity], dtype=np.float64)
            for k_value, spectrum_value in zip(k_values, values, strict=True):
                self._writer.writerow(
                    {
                        "time": float(time),
                        "step": int(step),
                        "quantity": quantity,
                        self.k_column: float(k_value),
                        "value": float(spectrum_value),
                    }
                )
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class FullFieldHDF5Writer:
    """Write one HDF5 file per full real-space snapshot."""

    def __init__(
        self,
        path: str | Path,
        *,
        grid: Any,
        backend: Any,
        field_names: list[str],
        backend_name: str,
    ) -> None:
        self.path = Path(path)
        try:
            self._h5py = importlib.import_module("h5py")
        except ImportError as exc:
            raise ImportError(
                "Full-field output requires `h5py`. Install `h5py` or disable full-field "
                "output by setting t_out_full = 0."
            ) from exc

        self.path.mkdir(parents=True, exist_ok=True)
        self._next_index = 0
        self._metadata_attrs = {
            "Nx": int(grid.Nx),
            "Ny": int(grid.Ny),
            "Nz": int(grid.Nz),
            "Lx": float(grid.Lx),
            "Ly": float(grid.Ly),
            "Lz": float(grid.Lz),
            "backend": backend_name,
            "real_dtype": str(grid.real_dtype),
            "complex_dtype": str(grid.complex_dtype),
        }
        self._metadata_arrays = {
            "x": backend.to_numpy(grid.x),
            "y": backend.to_numpy(grid.y),
            "z": backend.to_numpy(grid.z),
            "field_names": np.asarray(field_names, dtype=self._h5py.string_dtype(encoding="utf-8")),
        }

    def snapshot_path(self, output_index: int) -> Path:
        """Return the on-disk path for a snapshot index."""

        return self.path / f"fullfield_{output_index + 1:04d}.h5"

    def write_state(
        self,
        state: Any,
        *,
        time: float,
        step: int,
        fft: Any,
        backend: Any,
        field_names: list[str] | None = None,
    ) -> int:
        output_index = self._next_index
        snapshot_path = self.snapshot_path(output_index)
        fields = extract_full_fields(
            state,
            fft,
            backend,
            field_names=state.field_names if field_names is None else field_names,
        )

        with self._h5py.File(snapshot_path, "w") as handle:
            metadata = handle.create_group("metadata")
            for key, value in self._metadata_attrs.items():
                metadata.attrs[key] = value
            for key, value in self._metadata_arrays.items():
                metadata.create_dataset(key, data=value)

            output_group = handle.create_group("output")
            output_group.create_dataset("time", data=np.asarray(float(time), dtype=np.float64))
            output_group.create_dataset("step", data=np.asarray(int(step), dtype=np.int64))
            for name, array in fields.items():
                output_group.create_dataset(name, data=np.asarray(array))

        self._next_index += 1
        return output_index

    def close(self) -> None:
        return None

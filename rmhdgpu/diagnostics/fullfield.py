"""Minimal helpers for extracting full real-space fields."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def extract_full_fields(
    state: Any,
    fft: Any,
    backend: Any,
    field_names: Iterable[str] | None = None,
    extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Inverse transform requested state fields and return NumPy arrays.

    `extra_fields` are already-computed real-space arrays (e.g. derived
    diagnostics such as Elsasser magnitudes) that are stored alongside the
    inverse-transformed state fields without any further transform.
    """

    names = list(field_names) if field_names is not None else state.field_names
    result = {name: backend.to_numpy(fft.c2r(state[name])) for name in names}
    if extra_fields:
        for name, array in extra_fields.items():
            result[name] = backend.to_numpy(array)
    return result


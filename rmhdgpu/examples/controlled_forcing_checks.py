"""Run the fixed target/power examples and check their physical work budgets.

Use a compute allocation on a cluster. Detailed numerical output stays in the
requested output directory; REPORT.json contains compact acceptance evidence.
These checks are separate from lightweight pytest tests and do not tune inputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from rmhdgpu.run import run_simulation
from rmhdgpu.runfile import resolve_run_settings


def read_rows(path):
    with path.open(encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def column(rows, name):
    return np.asarray([float(row[name]) for row in rows])


def run_check(input_path, output, backend):
    settings = resolve_run_settings(runfile_path=input_path, cli_overrides={
        "backend": {"backend": backend}, "output_dir": str(output),
    })
    config = settings.config
    endpoint_fields = {}

    def observe(event, **sample):
        if event == "post_forcing" and sample["time"] >= config.tmax:
            device = sample["backend"]
            endpoint_fields.update({name: device.scalar_to_float(
                device.xp.max(device.xp.abs(sample["state"][name])))
                for name in sample["state"].field_names})

    run_simulation(settings, observer=observe)
    rows = read_rows(output / "scalar_diagnostics.csv")
    events = read_rows(output / "forcing_events.csv")
    time = column(rows, "time")
    total = column(rows, "total_energy")
    work = sum(column(rows, f"forcing_{branch}_cumulative_work_z")
               for branch in config.controlled_shell.branches)
    residual = total - total[0] - work
    event_work = float(np.sum(column(events, "work_z")))
    kick_error = abs(event_work - work[-1]) / total[0]
    budget_error = float(np.max(abs(residual)) / total[0])
    cap_count = int(np.sum(column(events, "rate_cap_active")))
    floor_count = int(np.sum(column(events, "floor_active")))
    checks = {
        "exact_endpoint": bool(time[-1] == config.tmax),
        "final_interval_flushed": float(events[-1]["time"]) == config.tmax,
        "pending_interval_zero": float(rows[-1]["forcing_elapsed"]) == 0.0,
        "budget_residual_below_1e_minus3_initial": budget_error < 1e-3,
        "kick_sum_below_2e_minus12_initial": bool(kick_error < 2e-12),
        "no_cap_or_floor_events": cap_count == floor_count == 0,
        "finite_energy_and_work": bool(np.isfinite(total).all() and np.isfinite(work).all()),
        "endpoint_fields_finite": bool(endpoint_fields and all(np.isfinite(v) for v in endpoint_fields.values())),
    }
    statistics = {
        "time_final": float(time[-1]), "steps": int(rows[-1]["step"]),
        "event_rows": len(events), "startup_energy": float(total[0]),
        "final_energy": float(total[-1]), "actual_work": float(work[-1]),
        "max_budget_residual_over_initial_energy": budget_error,
        "kick_sum_error_over_initial_energy": float(kick_error),
        "rate_cap_events": cap_count, "floor_events": floor_count,
        "endpoint_field_max_abs": endpoint_fields,
    }
    for branch in config.controlled_shell.branches:
        selected = [row for row in events if row["branch"] == branch]
        spec = getattr(config.controlled_shell, branch)
        physical = column(rows, f"forcing_{branch}_total_energy_z")
        independent = column(rows, f"elsasser_energy_{branch}") / 2
        norm_error = float(np.max(abs(physical - independent)) / np.max(physical))
        checks[f"{branch}_normalization_below_3e_minus14"] = norm_error < 3e-14
        statistics[f"{branch}_normalization_error"] = norm_error
        statistics[f"{branch}_final_rms"] = float(column(rows, f"forcing_{branch}_total_rms_z")[-1])
        if spec.control == "target":
            target = spec.target_value**2 / 4
            ratio = column(selected, "total_post_z") / target
            error = -np.log(ratio) + np.log(spec.startup_energy_fraction) * np.exp(-column(selected, "time") / spec.tau_F)
            checks[f"{branch}_log_relaxation_below_1e_minus4"] = bool(np.max(abs(error)) < 1e-4)
            checks[f"{branch}_final_energy_above_0p995_target"] = bool(ratio[-1] > 0.995)
            statistics[f"{branch}_max_log_relaxation_error"] = float(np.max(abs(error)))
            statistics[f"{branch}_final_energy_over_target"] = float(ratio[-1])
        else:
            expected = spec.epsilon * column(selected, "interval")
            power_error = float(np.max(abs(column(selected, "work_z") - expected) / expected))
            checks[f"{branch}_each_power_event_relative_error_below_2e_minus11"] = power_error < 2e-11
            statistics[f"{branch}_max_power_event_relative_error"] = power_error
    if config.equation_set == "s09":
        minus = float(np.max(column(rows, "forcing_minus_total_rms_z")))
        checks["unforced_minus_below_1e_minus12_rms"] = minus < 1e-12
        checks["compressive_entropy_endpoint_exactly_zero"] = all(
            endpoint_fields.get(name) == 0.0 for name in ("upar", "dbpar", "s"))
        statistics["max_minus_rms"] = minus
    return {
        "accepted": all(checks.values()), "checks": checks, "statistics": statistics,
        "backend": backend, "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "resolved_input_sha256": hashlib.sha256((output / "resolved_config.toml").read_bytes()).hexdigest(),
        "scope": "Fixed nondissipative example; this is not a stationarity or resolution study.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("numpy", "scipy_cpu", "cupy"), default="scipy_cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    reports = {}
    for label in ("controlled_target_s09", "controlled_power_alfvenic"):
        reports[label] = run_check(root / "examples" / f"{label}.input", output / label, args.backend)
        (output / "REPORT.json").write_text(json.dumps(reports, indent=2) + "\n")
        if not reports[label]["accepted"]:
            raise SystemExit(f"{label} failed; inspect {output / 'REPORT.json'}")
    print(json.dumps({name: report["accepted"] for name, report in reports.items()}))


if __name__ == "__main__":
    main()

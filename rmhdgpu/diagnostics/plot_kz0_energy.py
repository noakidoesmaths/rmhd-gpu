import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV = HERE / "kz_scan" / "runs" / "kz_01" / "scalar_diagnostics.csv"
OUT = HERE / "kz_scan" / "kz1.png"

t, E = [], []
with CSV.open() as fh:
    for row in csv.DictReader(fh):
        t.append(float(row["time"]))
        E.append(float(row["total_energy"]))
t = np.asarray(t); E = np.asarray(E)

# Theory growth rate at k_par = 0 (interchange)
vA, chi, g, K_rho0 = 1.0, 0.03, 9.3, -13.0
kx_idx, ky_idx = 3, 2
N_sq = -g * (K_rho0 + g / (vA**2 * (1 + chi)))
ky2_over_kperp2 = ky_idx**2 / (kx_idx**2 + ky_idx**2)
gamma_th = float(np.sqrt(max(ky2_over_kperp2 * N_sq, 0.0)))

# log(E) = 2*gamma*t + const
fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
mask = E > 0.0
ax.plot(t[mask], np.log(E[mask]), lw=1.5, label="log(total_energy)")

# Linear regression over the fit window [15, 18] -- this is what
# scan_inhomo_rmhd_growth.py uses to extract gamma.
FIT_TMIN, FIT_TMAX = 16.0, 19.0
fit_mask = (t >= FIT_TMIN) & (t <= FIT_TMAX) & mask
slope, intercept = np.polyfit(t[fit_mask], np.log(E[fit_mask]), 1)
gamma_fit = 0.5 * slope

# Draw the regression line a bit beyond the fit window so the slope is visible.
t_line = np.linspace(16, 19, 100)
# ax.plot(t_line, slope * t_line + intercept,
#        "r-", ls = '--', lw=2.0,
#        label=fr"fit on [{FIT_TMIN:g}, {FIT_TMAX:g}]: $2\gamma_{{fit}} = {slope:.3f}$")
# ax.plot(t[fit_mask], np.log(E[fit_mask]), "ro", ms=4, label="fit points")

# Theory reference slope, anchored to the midpoint of the fit window.
t_mid = 0.5 * (FIT_TMIN + FIT_TMAX)
logE_mid = slope * t_mid + intercept
# ax.plot(t_line, logE_mid + 2 * gamma_th * (t_line - t_mid),
#        "--", color="0.4", lw=1.5,
#        label=fr"theory slope $2\gamma_{{theory}} = {2*gamma_th:.3f}$")

ax.axvspan(FIT_TMIN, FIT_TMAX, color="0.9")
ax.set_xlabel("t"); ax.set_ylabel("total_energy")
ax.set_title(f"kz_index = 0, (fourier mode [4,2,0])"); ax.grid(alpha=0.3)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150)
print(f"saved {OUT}")


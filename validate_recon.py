"""
Validation script for LMR reconstruction results.

Loads reconstruction via cfr.ReconRes(), compares against GISTEMP instrumental
observations, and produces spatial correlation maps, GMST time series plots,
and summary metrics.

Run inside davidedge/lmr2:latest Docker container.
"""

import os
import csv
import numpy as np
import xarray as xr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import cfr

# ── Configuration ────────────────────────────────────────────────────────────
RECON_DIR   = os.environ.get('RECON_DIR', '/recons')
OUT_DIR     = os.environ.get('VALIDATION_DIR', '/validation')
VALID_START = 1880
VALID_END   = 2000
ANOM_PERIOD = [1951, 1980]

os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load reconstruction ──────────────────────────────────────────────────
print(f'Loading reconstruction from {RECON_DIR} ...')
res = cfr.ReconRes(RECON_DIR)
res.load(['tas', 'tas_gm'], verbose=True)

recon_tas = res.recons['tas']      # ClimateField (ensemble-mean spatial field)
recon_gm  = res.recons['tas_gm']   # EnsTS (global mean, full ensemble)

# ── 2. Fetch GISTEMP observations ───────────────────────────────────────────
print('Fetching GISTEMP observations ...')
obs = cfr.ClimateField().fetch('gistemp1200_ERSSTv4')
obs = obs.get_anom(ref_period=ANOM_PERIOD)
obs = obs.annualize(months=list(range(1, 13)))

# ── 3. Spatial correlation map ──────────────────────────────────────────────
print(f'Computing spatial correlation ({VALID_START}-{VALID_END}) ...')
corr_field = recon_tas.compare(obs, stat='corr', timespan=[VALID_START, VALID_END])
geo_mean_corr = float(corr_field.geo_mean())

print(f'  Geographic mean correlation: {geo_mean_corr:.4f}')

# Plot spatial correlation
fig, ax = plt.subplots(1, 1, figsize=(12, 6),
                       subplot_kw={'projection': ccrs.Robinson()})
corr_field.plot(ax=ax,
                cbar_orientation='horizontal',
                cbar_pad=0.08,
                cbar_kwargs={'label': 'Correlation (r)', 'shrink': 0.7},
                projection=ccrs.Robinson(),
                latlon_range=(-90, 90, 0, 360),
                clim=(-1, 1),
                cmap='RdYlBu_r')
ax.set_title(f'Reconstruction vs GISTEMP ({VALID_START}-{VALID_END})\n'
             f'Geographic Mean r = {geo_mean_corr:.3f}', fontsize=13)
fig.savefig(os.path.join(OUT_DIR, 'spatial_corr_map.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)

# ── 4. GMST time series with ensemble spread ────────────────────────────────
print('Generating GMST time series plot ...')

# Obs global mean for overlay
obs_gm = obs.geo_mean()

fig, ax = plt.subplots(figsize=(14, 5))
recon_gm.plot_qs(ax=ax, color='steelblue', alpha=0.3,
                 label='Reconstruction (5-95% range)')
ax.plot(recon_gm.time, recon_gm.median, color='steelblue', lw=1.5,
        label='Reconstruction (median)')

# Overlay GISTEMP
ax.plot(obs_gm.time, obs_gm.value, color='red', lw=1.5,
        label='GISTEMP', alpha=0.8)

ax.set_xlabel('Year CE')
ax.set_ylabel('Temperature Anomaly (\u00b0C)')
ax.set_title('Global Mean Surface Temperature')
ax.legend(loc='upper left')
ax.set_xlim(0, 2000)
fig.savefig(os.path.join(OUT_DIR, 'gmst_timeseries.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)

# ── 5. Compute GMST correlation in instrumental period ──────────────────────
recon_time = np.array(recon_gm.time)
recon_vals = np.array(recon_gm.median)
obs_time   = np.array(obs_gm.time)
obs_vals   = np.array(obs_gm.value)

# Find common years in validation window
mask_r = (recon_time >= VALID_START) & (recon_time <= VALID_END)
mask_o = (obs_time >= VALID_START) & (obs_time <= VALID_END)
common_years = np.intersect1d(recon_time[mask_r].astype(int),
                              obs_time[mask_o].astype(int))

r_vals = np.array([recon_vals[recon_time.astype(int) == y][0] for y in common_years])
o_vals = np.array([obs_vals[obs_time.astype(int) == y][0] for y in common_years])

gmst_corr = float(np.corrcoef(r_vals, o_vals)[0, 1])
print(f'  GMST correlation ({VALID_START}-{VALID_END}): {gmst_corr:.4f}')

# ── 6. Count ensemble info ──────────────────────────────────────────────────
n_ens = len(recon_gm.value[0]) if hasattr(recon_gm.value[0], '__len__') else 1

# ── 7. Save metrics CSV ─────────────────────────────────────────────────────
metrics_path = os.path.join(OUT_DIR, 'validation_metrics.csv')
with open(metrics_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['metric', 'value'])
    writer.writerow(['geo_mean_spatial_corr', f'{geo_mean_corr:.4f}'])
    writer.writerow(['gmst_corr', f'{gmst_corr:.4f}'])
    writer.writerow(['validation_period', f'{VALID_START}-{VALID_END}'])
    writer.writerow(['anom_ref_period', f'{ANOM_PERIOD[0]}-{ANOM_PERIOD[1]}'])

# ── 8. Generate HTML index ──────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html>
<head>
  <title>LMR Validation Results</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           max-width: 960px; margin: 0 auto; padding: 24px; color: #1a1a1a; }}
    h1 {{ border-bottom: 2px solid #2563eb; padding-bottom: 8px; }}
    table {{ border-collapse: collapse; margin: 16px 0; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px 16px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    img {{ max-width: 100%; margin: 12px 0; border: 1px solid #e5e7eb; }}
    .back {{ margin-top: 24px; }}
  </style>
</head>
<body>
  <h1>LMR Reconstruction Validation</h1>

  <h2>Summary Metrics</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Spatial Correlation (geographic mean)</td><td>{geo_mean_corr:.4f}</td></tr>
    <tr><td>GMST Correlation ({VALID_START}-{VALID_END})</td><td>{gmst_corr:.4f}</td></tr>
    <tr><td>Anomaly Reference Period</td><td>{ANOM_PERIOD[0]}-{ANOM_PERIOD[1]}</td></tr>
  </table>

  <h2>Spatial Correlation Map</h2>
  <p>Pearson correlation between reconstruction and GISTEMP at each grid cell
     over {VALID_START}-{VALID_END}.</p>
  <img src="spatial_corr_map.png" alt="Spatial correlation map">

  <h2>Global Mean Surface Temperature</h2>
  <p>Reconstruction ensemble spread (5-95%) with GISTEMP overlay.</p>
  <img src="gmst_timeseries.png" alt="GMST time series">

  <p class="back"><a href="../index.html">&larr; Back to results</a></p>
</body>
</html>"""

with open(os.path.join(OUT_DIR, 'index.html'), 'w') as f:
    f.write(html)

print(f'\nValidation complete. Outputs in {OUT_DIR}/')

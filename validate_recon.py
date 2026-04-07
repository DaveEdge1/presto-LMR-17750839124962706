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
import cartopy.feature as cfeature

import cfr

# ── Configuration ────────────────────────────────────────────────────────────
RECON_DIR   = os.environ.get('RECON_DIR', '/recons')
OUT_DIR     = os.environ.get('VALIDATION_DIR', '/validation')
VALID_START = 1880
VALID_END   = 2000
ANOM_PERIOD = [1951, 1980]

os.makedirs(OUT_DIR, exist_ok=True)


def area_weighted_mean(da):
    """Area-weighted spatial mean of a DataArray with lat/lon dims."""
    wgts = np.cos(np.deg2rad(da['lat']))
    return float(da.weighted(wgts).mean(('lat', 'lon')).values)


def ensts_to_1d(ensts):
    """Extract a 1D time series from an EnsTS (uses median across ensemble)."""
    time = np.asarray(ensts.time)
    val = np.asarray(ensts.value)
    if val.ndim == 2:
        val_1d = np.nanmedian(val, axis=1)
    else:
        val_1d = val
    return time, val_1d


# ── 1. Load reconstruction ──────────────────────────────────────────────────
print(f'Loading reconstruction from {RECON_DIR} ...')
res = cfr.ReconRes(RECON_DIR)
res.load(['tas', 'tas_gm'], verbose=True)

recon_tas = res.recons['tas']      # ClimateField (ensemble-mean spatial field)
recon_gm  = res.recons['tas_gm']   # EnsTS (global mean, full ensemble)

# ── 2. Fetch GISTEMP observations ───────────────────────────────────────────
print('Fetching GISTEMP observations ...')
obs = cfr.ClimateField().fetch('gistemp1200_ERSSTv4', vn='tempanomaly')
obs = obs.get_anom(ref_period=ANOM_PERIOD)
obs = obs.annualize(months=list(range(1, 13)))

# ── 3. Spatial correlation map ──────────────────────────────────────────────
print(f'Computing spatial correlation ({VALID_START}-{VALID_END}) ...')
corr_field = recon_tas.compare(obs, stat='corr', timespan=[VALID_START, VALID_END])

# Compute geographic mean correlation manually from the underlying DataArray.
# compare() returns a ClimateField wrapping a 2D (lat, lon) correlation map.
corr_da = corr_field.da
geo_mean_corr = area_weighted_mean(corr_da)
print(f'  Geographic mean correlation: {geo_mean_corr:.4f}')

# Plot spatial correlation using matplotlib directly for reliability
fig, ax = plt.subplots(1, 1, figsize=(12, 6),
                       subplot_kw={'projection': ccrs.Robinson()})
corr_da.plot(ax=ax, transform=ccrs.PlateCarree(),
             cmap='RdYlBu_r', vmin=-1, vmax=1,
             cbar_kwargs={'label': 'Correlation (r)',
                          'orientation': 'horizontal',
                          'shrink': 0.7, 'pad': 0.08})
ax.coastlines(linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
ax.set_global()
ax.set_title(f'Reconstruction vs GISTEMP ({VALID_START}-{VALID_END})\n'
             f'Geographic Mean r = {geo_mean_corr:.3f}', fontsize=13)
fig.savefig(os.path.join(OUT_DIR, 'spatial_corr_map.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)

# ── 4. GMST time series with ensemble spread ────────────────────────────────
print('Generating GMST time series plot ...')

# recon_gm is an EnsTS: time shape (nt,), value shape (nt, nEns)
recon_time = np.asarray(recon_gm.time)
recon_val  = np.asarray(recon_gm.value)  # (nt, nEns)
if recon_val.ndim == 1:
    recon_val = recon_val.reshape(-1, 1)

recon_median = np.nanmedian(recon_val, axis=1)
recon_q05    = np.nanquantile(recon_val, 0.05, axis=1)
recon_q95    = np.nanquantile(recon_val, 0.95, axis=1)

# Obs global mean for overlay
obs_gm = obs.geo_mean()  # EnsTS
obs_time, obs_1d = ensts_to_1d(obs_gm)

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(recon_time, recon_q05, recon_q95,
                alpha=0.3, color='steelblue',
                label='Reconstruction (5-95% range)')
ax.plot(recon_time, recon_median, color='steelblue', lw=1.5,
        label='Reconstruction (median)')
ax.plot(obs_time, obs_1d, color='red', lw=1.5,
        label='GISTEMP', alpha=0.85)

ax.set_xlabel('Year CE')
ax.set_ylabel('Temperature Anomaly (\u00b0C)')
ax.set_title('Global Mean Surface Temperature')
ax.legend(loc='upper left')
ax.set_xlim(0, 2000)
ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
fig.savefig(os.path.join(OUT_DIR, 'gmst_timeseries.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)

# ── 5. Compute GMST correlation in instrumental period ──────────────────────
recon_years = recon_time.astype(int)
obs_years   = obs_time.astype(int)
common = np.intersect1d(recon_years, obs_years)
common = common[(common >= VALID_START) & (common <= VALID_END)]

if len(common) >= 5:
    r_vals = np.array([recon_median[recon_years == y][0] for y in common])
    o_vals = np.array([obs_1d[obs_years == y][0] for y in common])
    mask = np.isfinite(r_vals) & np.isfinite(o_vals)
    if mask.sum() >= 5:
        gmst_corr = float(np.corrcoef(r_vals[mask], o_vals[mask])[0, 1])
    else:
        gmst_corr = float('nan')
else:
    gmst_corr = float('nan')
print(f'  GMST correlation ({VALID_START}-{VALID_END}): {gmst_corr:.4f}')

# ── 6. Save metrics CSV ─────────────────────────────────────────────────────
metrics_path = os.path.join(OUT_DIR, 'validation_metrics.csv')
with open(metrics_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['metric', 'value'])
    writer.writerow(['geo_mean_spatial_corr', f'{geo_mean_corr:.4f}'])
    writer.writerow(['gmst_corr', f'{gmst_corr:.4f}'])
    writer.writerow(['validation_period', f'{VALID_START}-{VALID_END}'])
    writer.writerow(['anom_ref_period', f'{ANOM_PERIOD[0]}-{ANOM_PERIOD[1]}'])
    writer.writerow(['n_ensemble_members', int(recon_val.shape[1])])

# ── 7. Generate HTML index ──────────────────────────────────────────────────
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
    <tr><td>Ensemble Members</td><td>{int(recon_val.shape[1])}</td></tr>
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


#%%
import numpy as np
import json
import matplotlib.pyplot as plt

exp_name = 'statues'

if exp_name == 'stonehenge':
    r = 0.0289    # For Stonehenge
else:
    r = 0.046   # For Statues and Flightroom

vol_min = -12

# Load in data across all trials
data_catnips = {}
data_baseline = {}
data_nerfnav = {}

# FOR CATNIPS
sigmas = ['95', '99']
vmaxs = ['5e-6', '1e-6', '1e-7']
for sigma in sigmas:
    for vmax in vmaxs:
        save_fp = f'results_processed/catnips/{exp_name}/{sigma}_{vmax}'
        with open(save_fp + '/data.json', 'r') as f:
            meta = json.load(f)

        data_catnips[f'{vmax}({sigma} %)'] = meta

# FOR BASELINE
cutoffs = ['1e4', '1e3', '1e2']
for cutoff in cutoffs:
    save_fp = f'results_processed/baseline/{exp_name}/{cutoff}'
    with open(save_fp + '/data.json', 'r') as f:
        meta = json.load(f)

    data_baseline[cutoff] = meta

# FOR NERF-NAV
penaltys = ['1e2', '1e3', '1e4']
for penalty in penaltys:
    save_fp = f'results_processed/nerf-nav/{exp_name}/{penalty}'
    with open(save_fp + '/data.json', 'r') as f:
        meta = json.load(f)

    data_nerfnav[penalty] = meta

metrics = ['SDF', 'Vol. Int.']

fig, ax = plt.subplots(len(metrics), figsize=(8, 8), dpi=80)

counter = 0
for key, value in data_catnips.items():
    # Value is list of list containing all trajectories and all points in trajectories
    sdfs = value['sdfs']
    vols = value['vols']

    min_sdfs = np.array([np.array(sdf).min()-r for sdf in sdfs])
    max_vols = np.array([np.array(vol).max() for vol in vols])

    sdf_mean = np.mean(min_sdfs)
    vol_mean = np.mean(max_vols)
    if vol_mean == 0:
        vol_mean = vol_min
    else:
        vol_mean = np.log10(vol_mean)

    sdf_errors = np.array([sdf_mean - min_sdfs.min(), min_sdfs.max() - sdf_mean]).reshape(-1, 1)

    if max_vols.max() == 0:
        vol_errors = np.array([100, 0]).reshape(-1, 1)
    else:
        vol_errors = np.array([100, np.log10(max_vols.max()) - vol_mean]).reshape(-1, 1)

    # p0 = ax[0].bar(counter, sdf_mean, align='center', alpha=0.5, ecolor='black', capsize=10, yerr=sdf_errors)
    # p1 = ax[1].bar(counter, vol_mean, align='center', alpha=0.5, ecolor='black', capsize=10, yerr=vol_errors)

    ax[0].errorbar(counter, sdf_mean, yerr=sdf_errors, fmt="o", color="g", capsize=10)
    ax[1].errorbar(counter, vol_mean, yerr=vol_errors, fmt="o", color="g", capsize=10)

    counter += 1

for key, value in data_baseline.items():
    # Value is list of list containing all trajectories and all points in trajectories
    sdfs = value['sdfs']
    vols = value['vols']

    min_sdfs = np.array([np.array(sdf).min()-r for sdf in sdfs])
    max_vols = np.array([np.array(vol).max() for vol in vols])

    sdf_mean = np.mean(min_sdfs)
    vol_mean = np.mean(max_vols)
    if vol_mean == 0:
        vol_mean = vol_min
    else:
        vol_mean = np.log10(vol_mean)

    sdf_errors = np.array([sdf_mean - min_sdfs.min(), min_sdfs.max() - sdf_mean]).reshape(-1, 1)

    if max_vols.max() == 0:
        vol_errors = np.array([100, 0]).reshape(-1, 1)
    else:
        vol_errors = np.array([100, np.log10(max_vols.max()) - vol_mean]).reshape(-1, 1)

    # p0 = ax[0].bar(counter, sdf_mean, align='center', alpha=0.5, ecolor='black', capsize=10, yerr=sdf_errors)
    # p1 = ax[1].bar(counter, vol_mean, align='center', alpha=0.5, ecolor='black', capsize=10, yerr=vol_errors)

    ax[0].errorbar(counter, sdf_mean, yerr=sdf_errors, fmt="o", color="b", capsize=10)
    ax[1].errorbar(counter, vol_mean, yerr=vol_errors, fmt="o", color="b", capsize=10)

    counter += 1

for key, value in data_nerfnav.items():
    # Value is list of list containing all trajectories and all points in trajectories
    sdfs = value['sdfs']
    vols = value['vols']

    min_sdfs = np.array([np.array(sdf).min()-r for sdf in sdfs])
    max_vols = np.array([np.array(vol).max() for vol in vols])

    sdf_mean = np.mean(min_sdfs)
    vol_mean = np.mean(max_vols)
    if vol_mean == 0:
        vol_mean = vol_min
    else:
        vol_mean = np.log10(vol_mean)

    sdf_errors = np.array([sdf_mean - min_sdfs.min(), min_sdfs.max() - sdf_mean]).reshape(-1, 1)

    if max_vols.max() == 0:
        vol_errors = np.array([100, 0]).reshape(-1, 1)
    else:
        vol_errors = np.array([100, np.log10(max_vols.max()) - vol_mean]).reshape(-1, 1)

    # p0 = ax[0].bar(counter, sdf_mean, align='center', alpha=0.5, ecolor='black', capsize=10, yerr=sdf_errors)
    # p1 = ax[1].bar(counter, vol_mean, align='center', alpha=0.5, ecolor='black', capsize=10, yerr=vol_errors)

    ax[0].errorbar(counter, sdf_mean, yerr=sdf_errors, fmt="o", color="m", capsize=10)
    ax[1].errorbar(counter, vol_mean, yerr=vol_errors, fmt="o", color="m", capsize=10)

    counter += 1

#%%
ax[0].axhline(y = 0., color = 'r', linestyle = '--', alpha=0.3) 
ax[0].set_ylabel('SDF')
ax[1].set_ylabel('Volume Intersection')
ax[1].set_ylim([vol_min, -4.])
#%%
# plt.show()
#%%
plt.savefig(f'./results_processed/statistics_{exp_name}.png')
# %%

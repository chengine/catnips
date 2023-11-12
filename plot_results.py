
#%%
import numpy as np
import json
import matplotlib.pyplot as plt

exp_name = ['statues']
methods = ['catnips']
sigmas = ['95', '99']
vmaxs = ['5e-6', '1e-6', '1e-7']

# Load in data across all trials
data_catnips = {}
data_baseline = {}
data_nerfnav = {}

for exp in exp_name:
    for method in methods:    
        if method == 'catnips':
            for sigma in sigmas:
                for vmax in vmaxs:
                    save_fp = f'results_processed/{method}/{exp}/{sigma}_{vmax}'

                    with open(save_fp + '/data.json', 'r') as f:
                        meta = json.load(f)

                    data_catnips[sigma + '_' + vmax] = meta

metrics = ['SDF', 'Vol. Int.']

fig, ax = plt.subplots(len(metrics), figsize=(8, 8), dpi=80)

counter = 0
for key, value in data_catnips.items():
    # Value is list of list containing all trajectories and all points in trajectories
    sdfs = value['sdfs']
    vols = value['vols']

    min_sdfs = np.array([np.array(sdf).min() for sdf in sdfs])
    max_vols = np.array([np.array(vol).max() for vol in vols])

    sdf_mean = np.mean(min_sdfs)
    vol_mean = np.mean(max_vols)

    sdf_errors = np.array([sdf_mean - min_sdfs.min(), min_sdfs.max() - sdf_mean]).reshape(-1, 1)
    vol_errors = np.array([vol_mean - max_vols.min(), max_vols.max() - vol_mean]).reshape(-1, 1)

    p0 = ax[0].bar(key, sdf_mean, align='center', alpha=0.5, ecolor='black', capsize=10)
    p1 = ax[1].bar(key, vol_mean, align='center', alpha=0.5, ecolor='black', capsize=10)

    ax[0].errorbar(counter, sdf_mean, yerr=sdf_errors, fmt="o", color="r")
    ax[1].errorbar(counter, vol_mean, yerr=vol_errors, fmt="o", color="r")

    counter += 1

#%%
ax[0].set_ylabel('SDF')
ax[1].set_ylabel('Volume Intersection')

#%%
# plt.show()
#%%
plt.savefig('./results_processed/statistics.png')
# %%

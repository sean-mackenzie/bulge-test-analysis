from os.path import join
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from other_physics.bulge_equations import pressure_by_deflection_Rosset
from other_physics.calculate_thickness_after_pre_stretch import calculate_stretched_thickness


# ---

BASE_DIR = '/Users/mackenzie/Library/CloudStorage/Box-Box/2024/Bulge Tests'
SAVE_DIR = join(BASE_DIR, 'Analyses/overlay_bulge_test_data_on_bulge_equations')
FP_ALL_PZ = join(SAVE_DIR, 'custom_all-bulge-test-analyses-PZ-by-t-by-tid.xlsx')

# ---

# Define typical parameters
t = 20e-6
w_o = np.linspace(2, 225, 100) * 1e-6
E_MPa = 3.25  # np.array([1, 5, 10, 50]) * 1e6
mu = 0.5
sigma_o_kPa = np.array([28])

# read bulge test data: pressure vs. deflection
df = pd.read_excel(FP_ALL_PZ)
# df = df[(df['pre_stretch_measured'] > 12) & (df['pre_stretch_measured'] < 18)]
df = df[df['bta_id'] == '20250104_C9-0pT_20nmAu_4mmDia']

pre_stretch_model_ = 0.0  # df['pre_stretch_measured'].mean()
pre_stretch_model = np.round(1 + pre_stretch_model_ / 100, 3)
thickness_post_stretch = calculate_stretched_thickness(original_thickness=t, stretch_factor=pre_stretch_model)
thickness_post_stretch_ = np.round(thickness_post_stretch * 1e6, 2)

#SAVE_DIR = join(SAVE_DIR, 'E={}MPa'.format(E_MPa))
#if not os.path.exists(SAVE_DIR):
#    os.makedirs(SAVE_DIR)

# iterate
radii = df['radius_mm'].unique()
radii = [2.0]
# plot
for r in radii:
    df_r = df[df['radius_mm'] == r]

    fig, ax = plt.subplots(figsize=(8, 6))
    for bta_id in df_r.sort_values('pre_stretch_measured')['bta_id'].unique():
        df_bta = df_r[df_r['bta_id'] == bta_id]

        # get only data for tid of max pressure
        tid_of_max_pressure = df_bta['tid'].iloc[np.argmax(df_bta['P'])]
        df_bta_max_pressure = df_bta[df_bta['tid'] == tid_of_max_pressure]

        lbl_memb_id = df_bta_max_pressure['memb_id'].iloc[0]
        lbl_deposit_Au = df_bta_max_pressure['deposit_Au'].iloc[0]
        lbl_pre_stretch_measured = df_bta_max_pressure['pre_stretch_measured'].iloc[0]


        ax.plot(df_bta_max_pressure['z'], df_bta_max_pressure['P'], '-o', ms=2,
                label='{}: {}pT + {}nmAu'.format(lbl_memb_id, lbl_pre_stretch_measured, lbl_deposit_Au))

    for sigma_o in sigma_o_kPa:
        p = pressure_by_deflection_Rosset(w_o, sigma_o * 1e3, r * 1e-3, thickness_post_stretch, E_MPa * 1e6, mu)
        ax.plot(w_o * 1e6, p, '-', lw=1, label='Rossett (E={} MPa, sigma_o={} kPa)'.format(E_MPa, sigma_o))

    ax.set_xlabel('Deflection (um)')
    ax.set_ylabel('Pressure (Pa)')
    ax.grid(alpha=0.25)
    ax.legend(loc='best', fontsize='small', markerscale=3)  # , bbox_to_anchor=(1, 1)
    ax.set_title('Radius = {} mm: Pre-stretch={}, Post-stretch-thickness={}um'.format(r, pre_stretch_model, thickness_post_stretch_))

    ax.set_ylim([-10, 240])
    if r == 1.0:
        ax.set_ylim([-25, 625])
        ax.set_xlim([-2.5, 32.5])

    plt.tight_layout()
    plt.savefig(join(SAVE_DIR, 'E={}MPa_tmemb-PS={}um_r={}mm.png'.format(E_MPa, thickness_post_stretch_, r)),
                dpi=300, facecolor='w', bbox_inches='tight')
    plt.show()
    plt.close()
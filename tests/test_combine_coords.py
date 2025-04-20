from os.path import join
import os
import numpy as np
import pandas as pd
from utils import io, plotting

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    BULGE_ID = '20250105_C5-30pT_15nmAu_4mmDia'

    ROOT_DIR = '/Users/mackenzie/Library/CloudStorage/Box-Box/2024/Bulge Tests/Analyses'
    BASE_DIR = join(ROOT_DIR, BULGE_ID)
    SAVE_DIR = join(BASE_DIR, 'analyses')
    FDIR = join(BASE_DIR, 'results/coords')
    FTYPE = '.xlsx'
    SUBSTRING = 'test_coords_'
    SORT_STRINGS = ['test-', '.xlsx']
    SCALE_Z = 1  # may need to rescale z if not scaled during 3D particle tracking analysis
    FLIP_Z = False  # if positive pressure, True. If vacuum pressure, False.
    SKIP_FRAME_ZERO = True

    BEST_PIDS = [19]
    BAD_PIDS = []
    GOOD_PIDS = [14, 15, 12, 22, 19]
    GOOD_PIDS = [x for x in GOOD_PIDS if x not in BAD_PIDS]

    FRAME_RATE = 11.001
    START_FRAME = 16
    TIME_START = START_FRAME / FRAME_RATE  # seconds (may vary between tests): avg(z: t < time_start)
    TIME_EVAL = (185 / FRAME_RATE, 205 / FRAME_RATE)  # seconds (avg(dz: time_eval1 < t < time_eval2))
    Z0 = ('auto', START_FRAME)  # if tuple, z0 (zero deflection) = avg(z: t < start_frame)

    # FUNCTION: COMBINE COORDS
    save_dir_ = SAVE_DIR
    t_start = TIME_START
    t_eval = TIME_EVAL

    for ONLY_PIDS, lbl in zip([GOOD_PIDS, BEST_PIDS], ['good-pids', 'best-pids']):

        save_dir = join(save_dir_, 'figs', lbl)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        read_coords = True
        if read_coords:
            dfs = io.combine_coords(fdir=FDIR, substring=SUBSTRING, sort_strings=SORT_STRINGS,
                                    scale_z=SCALE_Z, z0=Z0, flip_z=FLIP_Z,
                                    frame_rate=FRAME_RATE, skip_frame_zero=SKIP_FRAME_ZERO,
                                    only_pids=ONLY_PIDS)
            dfs.to_excel(join(save_dir, 'combined_coords.xlsx'), index=False)
            plotting.show_combined_z_by_dt(dfs, savepath=join(save_dir, 'combined_coords.png'))
        else:
            dfs = pd.read_excel(join(save_dir_, 'combined_coords.xlsx'))

        # -

        # FUNCTION: PLOT PARTICLES THAT DEFLECT THE MOST

        # determine which particles deflect the most
        # since these will likely be in the center of the disc
        tids = dfs['tid'].unique()
        dfdzs = []
        for tid in tids:
            dft = dfs[dfs['tid'] == tid]

            df_init = dft[dft['dt'] < t_start]
            df_eval = dft[(dft['dt'] > t_eval[0]) & (dft['dt'] < t_eval[1])]

            df_init = df_init[['id', 'x', 'y', 'z']].groupby('id').mean()
            df_eval = df_eval[['id', 'x', 'y', 'z']].groupby('id').mean()

            dfdz = df_init.merge(right=df_eval, how='inner', left_index=True, right_index=True, suffixes=("_i", "_f"))
            dfdz['dz'] = dfdz['z_f'] - dfdz['z_i']
            dfdz = dfdz.sort_values('dz', ascending=False)
            dfdz['tid'] = tid
            dfdz = dfdz.reset_index()
            dfdzs.append(dfdz)

            pids = dfdz['id']
            num_pids = 10
            df = dft
            if pids is None:
                pids = df['id'].unique()
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for pid in pids[:num_pids]:
                dfpid = df[df['id'] == pid]
                ax.plot(dfpid['dt'], dfpid['z'], '-o', ms=1, linewidth=0.5,
                        label='{}: {}'.format(pid, np.round(dfpid['z'].max(), 1)),
                        )
            ax.set_xlabel('dt')
            ax.set_ylabel('z')
            ax.grid(alpha=0.25)
            ax.legend(loc='lower center', ncols=3, fontsize='small', title=r'$p_{ID}: \: \Delta_{max} z \: (\mu m)$')
            plt.tight_layout()
            plt.savefig(join(save_dir, 'max_dz_tid{}.png'.format(tid)))

        dfdz = pd.concat(dfdzs)
        dfdz.to_excel(join(save_dir_, 'combined_dz.xlsx'))
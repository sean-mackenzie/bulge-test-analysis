from os.path import join
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

if __name__ == '__main__':

    BASE_DIR = '/Users/mackenzie/Desktop/Bulge Test/Experiments/20250105_C9-0pT_NoMetal_3mmDia'
    READ_DIR = join(BASE_DIR, 'analyses')
    SAVE_DIR = READ_DIR
    FIGS_DIR = join(READ_DIR, 'figs')

    SAVE_PLOTS = True
    PLOT_PER_PID = False
    SAVE_DF = True

    ONLY_PIDS = [5]  # should be the same pid as 'best-pids', which was used to generate combined_dz.xlsx

    DICT_DT_OVERLAYS = {
        1: 12.25,
        2: 12.575,
        3: 12.35,
        4: 13.6,
    }

    # ---
    read_dir = READ_DIR
    save_dir = SAVE_DIR
    figs_dir = FIGS_DIR
    save_plots = SAVE_PLOTS
    save_df = SAVE_DF
    only_pids = ONLY_PIDS
    dict_dt_overlays = DICT_DT_OVERLAYS
    plot_per_pid = PLOT_PER_PID

    # ---

    for pth in [save_dir, figs_dir]:
        if not os.path.exists(pth):
            os.makedirs(pth)

    dfp = pd.read_excel(join(read_dir, 'combined_P_by_dt.xlsx'))
    dfz = pd.read_excel(join(read_dir, 'combined_coords.xlsx'))

    dfz = dfz[dfz['id'].isin(only_pids)]

    tids = dfz['tid'].unique()

    dfs = []
    for tid in tids:
        dfptid = dfp[dfp['tid'] == tid]
        dfztid = dfz[dfz['tid'] == tid]

        dfztid['dt_overlay'] = dfztid['dt'] + dict_dt_overlays[tid]  #  + dfptid['dt'].iloc[0]
        dfs.append(dfztid)
        dfztid['dt_rel'] = dfztid['dt_overlay']

        xlim_min = int(np.round(dfztid['dt_rel'].min() - 2.1))
        xlim_max = int(np.round(dfztid['dt_rel'].max() + 2.1))

        if save_plots:
            # fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(10, 6))
            fig, ax1 = plt.subplots(figsize=(10, 4))
            ax2 = ax1.twinx()

            ax1.plot(dfptid['dt'], dfptid['P'], 'r-o', ms=1, linewidth=0.5, zorder=3.5)
            ax1.set_xlabel('Time (s)')
            ax1.set_xlim([xlim_min - 0.5, xlim_max + 0.5])
            ax1.set_xticks(np.arange(xlim_min, xlim_max + 1))
            ax1.set_ylabel('Pressue (Pa)', color='r')
            ax1.grid(alpha=0.25)

            if plot_per_pid:
                for pid in dfztid['id'].unique():
                    dfpid = dfztid[dfztid['id'] == pid]
                    ax2.plot(dfpid['dt_rel'], dfpid['z'], 'o', ms=2, label=pid)  # , linewidth=0.5
            else:
                dfg = dfztid.groupby('dt_rel').mean().reset_index()
                ax2.plot(dfg['dt_rel'], dfg['z'], 'ko', ms=1, label=r'$\overline{z_{i}}$', zorder=3)  # , linewidth=0.5
            #ax2.set_xlabel('Time (s)')
            ax2.set_ylabel(r'$z \: (\mu m)$')
            ax2.legend(title=r'$p_{ID}$')
            # ax2.grid(alpha=0.25)
            plt.tight_layout()
            plt.savefig(join(figs_dir, 'tid{}.png'.format(tid)), dpi=300, facecolor='white')
            plt.close()

    if save_df:
        dfs = pd.concat(dfs)
        dfs.to_excel(join(save_dir, 'combined_coords_dt-aligned-to-pressure.xlsx'))
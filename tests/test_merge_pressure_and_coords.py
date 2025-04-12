from os.path import join
import os
import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

if __name__ == '__main__':

    BASE_DIR = '/Users/mackenzie/Library/CloudStorage/Box-Box/2024/Bulge Tests/Analyses/20250302_C17-20pT_25nmAu_2mmDia'
    READ_DIR = join(BASE_DIR, 'analyses')
    SAVE_DIR = READ_DIR
    FIGS_DIR = join(READ_DIR, 'figs')

    ONLY_TIDS = [5, 6, 7]
    ONLY_PIDS = None  # None = all pids, otherwise should be a subset of those in combined_coords.xlsx
    PLOT_PER_PID = False

    DICT_DT_OVERLAYS = {  # These values define where the black scatter points (IDPT) begin (t=0)
        1: 10.75,
        2: 6.8,
        3: 8.35,
        4: 6,
        5: 5.15,
        6: 7.65,
        7: 7.35,
    }

    # ---
    read_dir = READ_DIR
    save_dir = SAVE_DIR
    figs_dir = FIGS_DIR
    save_plots = True  # should always just be True
    save_df = True  # should always just be True
    only_pids = ONLY_PIDS
    dict_dt_overlays = DICT_DT_OVERLAYS
    plot_per_pid = PLOT_PER_PID

    # ---

    for pth in [save_dir, figs_dir]:
        if not os.path.exists(pth):
            os.makedirs(pth)

    dfp = pd.read_excel(join(read_dir, 'combined_P_by_dt.xlsx'))
    dfz = pd.read_excel(join(read_dir, 'combined_coords.xlsx'))

    if only_pids is not None:
        dfz = dfz[dfz['id'].isin(only_pids)]

    if ONLY_TIDS is None:
        tids = dfz['tid'].unique()
    else:
        tids = ONLY_TIDS

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
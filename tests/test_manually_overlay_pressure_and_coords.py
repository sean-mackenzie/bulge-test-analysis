from os.path import join
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

if __name__ == '__main__':

    BASE_DIR = '/Users/mackenzie/Desktop/Bulge Test/BulgeTest_200umSILP_4mmDia'
    SAVE_DIR = join(BASE_DIR, 'figs')

    dfp = pd.read_excel(join(BASE_DIR, 'combined_P_by_dt.xlsx'))
    dfz = pd.read_excel(join(BASE_DIR, 'combined_coords.xlsx'))
    tids = dfz['tid'].unique()

    dict_dt_overlays = {
        '1': 157.5,
        '2': 5.1,
        '3': 6,
        '4': 7,
        '5': 2.5,
        '6': 3,
        '7': 4,
        '8': 2,
        '9': 1,
        '10': 1,
        '11': 2.1,
        '12': 3,
        '13': 2.5,
        '14': 2.25,
        '15': 3,
        '16': 3.5,
        '17': 2.5,
        '18': 2,
    }

    for tid in [18]:
        dfptid = dfp[dfp['tid'] == tid]
        dfztid = dfz[dfz['tid'] == tid]

        dt_overlay = 2

        dfztid['dt_rel'] = dfztid['dt'] + dfptid['dt'].iloc[0] + dt_overlay

        fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(dfptid['dt'], dfptid['P'], 'k-o', ms=2, linewidth=1.5, zorder=3.5)
        ax.set_xlabel('Time (s)')
        ax.set_xlim([dfztid['dt_rel'].min() - 2.5, dfztid['dt_rel'].max() + 2.5])
        ax.set_xticks(np.arange(int(np.round(dfztid['dt_rel'].min() - 2, 0)), dfztid['dt_rel'].max() + 2, 1))
        ax.set_ylabel('Pressue (Pa)')
        ax.grid(alpha=0.25)

        ax2 = ax.twinx()
        for pid in [0]: # dfztid['id'].unique():
            dfpid = dfztid[dfztid['id'] == pid]
            ax2.plot(dfpid['dt_rel'], dfpid['z'], '-o', ms=1, zorder=2.5, label=pid)
        # ax2.set_xlabel('Time (s)')
        ax2.set_ylabel(r'$z \: (\mu m)$')
        ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$p_{ID}$')
        # ax2.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()
        # plt.savefig(join(SAVE_DIR, 'tid{}.png'.format(tid)))  # , dpi=300, facecolor='white')
        plt.close()
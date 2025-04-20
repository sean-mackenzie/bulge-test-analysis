import os
from os.path import join

import numpy as np

from utils import io, plotting, processing

if __name__ == '__main__':
    BULGE_ID = '20250105_C5-30pT_15nmAu_4mmDia'

    ROOT_DIR = '/Users/mackenzie/Library/CloudStorage/Box-Box/2024/Bulge Tests/Analyses'
    BASE_DIR = join(ROOT_DIR, BULGE_ID)
    SAVE_DIR = join(BASE_DIR, 'analyses')
    SAVE_FIG = join(SAVE_DIR, 'figs')
    FDIR = join(BASE_DIR, 'pressure')
    FTYPE = '.txt'
    FN_STR1 = 'test'
    PCOLS = ['t_reset', 'T', 'P']
    PRESSURE_CORRECTION_FUNCTION = -1

    # extrapolate
    EXTRAPOLATE_TIDS = None  # [4, 5]  # if None, then do not extrapolate
    FIT_DTS = [(16, 24)]  # (t_min, t_max): time points corresponding to pressure range to fit to
    EXTRAPOLATE_TOS = [453*2]  # pressure (Pa) to linearly extrapolate fitted line to

    for pth in [SAVE_DIR, SAVE_FIG]:
        if not os.path.exists(pth):
            os.makedirs(pth)

    if PRESSURE_CORRECTION_FUNCTION == -1:
        suptitle = "Pressure C.F. = -1 (vacuum pressure applied at test)".format(PRESSURE_CORRECTION_FUNCTION)
    elif PRESSURE_CORRECTION_FUNCTION == 1:
        suptitle = None
    else:
        suptitle = "Pressure Correction Function = {}".format(PRESSURE_CORRECTION_FUNCTION)

    dfs = io.combine_pressure_txt(fdir=FDIR, ftype=FTYPE, fn_strings=[FN_STR1, FTYPE], cols=PCOLS,
                                  correction_function=PRESSURE_CORRECTION_FUNCTION)

    # plot pressure profiles overlay
    plot_ = True
    if plot_:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 3.5))
        for tid in dfs['tid'].unique():
            dft = dfs[dfs['tid'] == tid]
            ax.plot(dft['dt'], dft['P'], '-', linewidth=0.5,
                    label='{}: {} Pa'.format(tid, dft['P'].max()))
        ax.set_xlabel('dt (s)')
        ax.set_xticks(np.arange(0, dfs['dt'].max(), 2))
        ax.set_ylabel('P (Pa)')
        ax.grid(alpha=0.25)
        ax.legend(title='TID: Pmax')
        plt.tight_layout()
        plt.savefig(join(SAVE_FIG, 'overlay-P_by_dt.png'), dpi=300, facecolor='white', bbox_inches='tight')
        plt.show()
        plt.close()


    # extrapolate pressure data
    if EXTRAPOLATE_TIDS is not None:
        # 1. export "raw" data
        dfs.to_excel(join(SAVE_DIR, 'combined_P_by_dt__raw.xlsx'), index=False)
        plotting.show_combined_P_by_dt(dfs, savepath=join(SAVE_FIG, 'combined_P_by_dt__raw.png'), suptitle=suptitle)

        for EXTRAPOLATE_TID, FIT_DT, EXTRAPOLATE_TO in zip(EXTRAPOLATE_TIDS, FIT_DTS, EXTRAPOLATE_TOS):
            dfs = processing.replace_with_extrapolated_pressure(
                dfs=dfs,
                extrapolate_tid=EXTRAPOLATE_TID,
                fit_dt=FIT_DT,
                extrapolate_to=EXTRAPOLATE_TO,
                savepath=join(SAVE_FIG, 'extrapolated_P_by_dt_tid{}.png'.format(EXTRAPOLATE_TID)))


    dfs.to_excel(join(SAVE_DIR, 'combined_P_by_dt.xlsx'), index=False)
    plotting.show_combined_P_by_dt(dfs, savepath=join(SAVE_FIG, 'combined_P_by_dt.png'), suptitle=suptitle)

    # ---

    # LEAK TEST
    evaluate_leak_test = False
    if evaluate_leak_test:
        FDIR = join(BASE_DIR, 'pressure', 'leak_test')
        FN_STR1 = 'pos_leak_test'
        SAVE_DIR = join(BASE_DIR, 'analyses', 'leak_test')
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

        dfs = io.combine_pressure_txt(fdir=FDIR, ftype=FTYPE, fn_strings=[FN_STR1, FTYPE], cols=PCOLS,
                                      correction_function=PRESSURE_CORRECTION_FUNCTION)
        dfs.to_excel(join(SAVE_DIR, 'leak_tests_P_by_dt.xlsx'), index=False)
        plotting.show_combined_P_by_dt(dfs, savepath=join(SAVE_DIR, 'leak_tests_P_by_dt.png'), suptitle=suptitle)
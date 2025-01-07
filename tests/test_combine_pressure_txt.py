import os
from os.path import join
from utils import io, plotting

if __name__ == '__main__':

    BASE_DIR = '/Users/mackenzie/Desktop/Bulge Test/Experiments/20250104_C7-20pT_4mmDia'
    SAVE_DIR = join(BASE_DIR, 'analyses')
    SAVE_FIG = join(SAVE_DIR, 'figs')
    FDIR = join(BASE_DIR, 'pressure')
    FTYPE = '.txt'
    FN_STR1 = 'test'
    PCOLS = ['t_reset', 'T', 'P']
    PRESSURE_CORRECTION_FUNCTION = 1

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
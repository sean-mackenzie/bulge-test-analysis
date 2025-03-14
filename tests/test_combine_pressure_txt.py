import os
from os.path import join
from utils import io, plotting, processing

if __name__ == '__main__':

    BASE_DIR = '/Users/mackenzie/Desktop/Bulge Test/Experiments/20250225_C13-20pT-25nmAu_2mmDia'
    SAVE_DIR = join(BASE_DIR, 'analyses')
    SAVE_FIG = join(SAVE_DIR, 'figs')
    FDIR = join(BASE_DIR, 'pressure')
    FTYPE = '.txt'
    FN_STR1 = 'test'
    PCOLS = ['t_reset', 'T', 'P']
    PRESSURE_CORRECTION_FUNCTION = 1

    # extrapolate
    EXTRAPOLATE_TID = None  # if None, then do not extrapolate
    FIT_DT = (9.5, 11.5)  # (t_min, t_max): time points corresponding to pressure range to fit to
    EXTRAPOLATE_TO = 1530  # pressure (Pa) to linearly extrapolate fitted line to

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

    # extrapolate pressure data
    if EXTRAPOLATE_TID is not None:
        # 1. export "raw" data
        dfs.to_excel(join(SAVE_DIR, 'combined_P_by_dt__raw.xlsx'), index=False)
        plotting.show_combined_P_by_dt(dfs, savepath=join(SAVE_FIG, 'combined_P_by_dt__raw.png'), suptitle=suptitle)

        dfs = processing.replace_with_extrapolated_pressure(
            dfs=dfs,
            extrapolate_tid=EXTRAPOLATE_TID,
            fit_dt=FIT_DT,
            extrapolate_to=EXTRAPOLATE_TO,
            savepath=join(SAVE_FIG, 'extrapolated_P_by_dt.png'))


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
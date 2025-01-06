from os.path import join
from utils import io, plotting

if __name__ == '__main__':

    BASE_DIR = '/Users/mackenzie/Desktop/Bulge Test/Experiments/20250105_C9-0pT_NoMetal_3mmDia'
    SAVE_DIR = join(BASE_DIR, 'analyses')
    FDIR = join(BASE_DIR, 'pressure')
    FTYPE = '.txt'
    FN_STR1 = 'test'
    PCOLS = ['t_reset', 'T', 'P']
    PRESSURE_CORRECTION_FUNCTION = 1

    dfs = io.combine_pressure_txt(fdir=FDIR, ftype=FTYPE, fn_strings=[FN_STR1, FTYPE], cols=PCOLS,
                                  correction_function=PRESSURE_CORRECTION_FUNCTION)
    dfs.to_excel(join(SAVE_DIR, 'combined_P_by_dt.xlsx'), index=False)
    plotting.show_combined_P_by_dt(dfs, savepath=join(SAVE_DIR, 'figs', 'combined_P_by_dt.png'))
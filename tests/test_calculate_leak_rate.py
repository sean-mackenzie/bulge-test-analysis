import os
from os.path import join
from utils import io, plotting, processing

if __name__ == '__main__':

    BASE_DIR = '/Users/mackenzie/Library/CloudStorage/Box-Box/2024/Bulge Tests/Analyses/20250104_C9-0pT_20nmAu_4mmDia'
    SAVE_DIR = join(BASE_DIR, 'analyze_leak_rate')
    SAVE_FIG = SAVE_DIR
    FDIR = join(BASE_DIR, 'pressure', 'leak rate')
    FTYPE = '.txt'
    FN_STR1 = 'leak_test'
    PCOLS = ['t_reset', 'T', 'P']
    PRESSURE_CORRECTION_FUNCTION = 1

    for pth in [SAVE_DIR, SAVE_FIG]:
        if not os.path.exists(pth):
            os.makedirs(pth)

    dfs = io.combine_pressure_txt(fdir=FDIR, ftype=FTYPE, fn_strings=[FN_STR1, FTYPE], cols=PCOLS,
                                  correction_function=PRESSURE_CORRECTION_FUNCTION,
                                  fn_startswith=FN_STR1,
                                  )
    import matplotlib.pyplot as plt

    px, py = 'dt', 'P'

    fig, ax = plt.subplots()
    for tid in dfs['tid'].unique():
        dft = dfs[dfs['tid'] == tid]
        ax.plot(dft[px], dft[py], '-o', ms=1, linewidth=0.5, label=tid)
    ax.set_xlabel('dt (s)')
    ax.set_ylabel('P (Pa)')
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(join(SAVE_FIG, f'combined-{FN_STR1}_P_by_dt.png'),
                dpi=300, facecolor='white', bbox_inches='tight')
    plt.show()
    plt.close()

    a = 1
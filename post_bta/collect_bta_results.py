from os.path import join
import os
import numpy as np
import pandas as pd


def get_measured_stretch(memb_casette_id):
    if memb_casette_id == 13:
        return 21.5
    elif memb_casette_id == 14:
        return 12.6
    elif memb_casette_id == 15:
        return 14.6
    elif memb_casette_id == 16:
        return 23
    elif memb_casette_id == 17:
        return 25.2
    elif memb_casette_id == 18:
        return 13.5
    elif memb_casette_id == 19:
        return 13.1
    elif memb_casette_id == 20:
        return 17
    elif memb_casette_id == 21:
        return 16
    elif memb_casette_id == 22:
        return 26
    else:
        return np.nan


BULGE_DIR = '/Users/mackenzie/Library/CloudStorage/Box-Box/2024/Bulge Tests/Analyses'
SUB_PATH_RESULTS_OVERVIEW = 'analyses/fit-w-by-p'
FN_PZ = 'PZ_by_t__tid{}.xlsx'
FN_RESULTS_OVERVIEW = 'results-overview.xlsx'


BULGE_TEST_DIRS = [x for x in os.listdir(BULGE_DIR) if x.startswith('2')]

super_dfpzs = []
super_df_results = []

for d in BULGE_TEST_DIRS:
    bulge_test_id = d
    test_date = d.split('_')[0]
    memb_id = d.split('_')[1]
    memb_cassette_id = int(memb_id.split('-')[0].split('C')[1])
    memb_pre_stretch_nominal = int(memb_id.split('-')[1].split('pT')[0])
    memb_pre_stretch_measured = get_measured_stretch(memb_cassette_id)
    thickness_Au = d.split('_')[2].split('nmAu')[0]
    bulge_diameter = float(d.split('_')[3].split('mmDia')[0])

    tid_dirs = [x for x in os.listdir(join(BULGE_DIR, bulge_test_id, SUB_PATH_RESULTS_OVERVIEW)) if x.startswith('tid')]
    tid_dirs.sort()
    dfpzs = []
    add_data = []
    for tid in tid_dirs:
        tid_num = int(tid.split('tid')[1])
        dfpz = pd.read_excel(join(BULGE_DIR, bulge_test_id, SUB_PATH_RESULTS_OVERVIEW, tid, FN_PZ.format(tid_num)))
        dfpz['tid'] = tid_num
        dfpzs.append(dfpz)
        add_data.append([tid_num, dfpz['P'].iloc[-1], dfpz['z'].iloc[-1]])
    dfpzs = pd.concat(dfpzs)
    dfpzs['bta_id'] = bulge_test_id
    dfpzs['test_date'] = test_date
    dfpzs['memb_id'] = memb_id
    dfpzs['memb_cid'] = memb_cassette_id
    dfpzs['pre_stretch_nominal'] = memb_pre_stretch_nominal
    dfpzs['pre_stretch_measured'] = memb_pre_stretch_measured
    dfpzs['thickness_Au'] = thickness_Au
    dfpzs['radius_mm'] = bulge_diameter / 2
    super_dfpzs.append(dfpzs)

    df_add = pd.DataFrame(add_data, columns=['tid', 'P_last_Pa', 'z_last_um']).set_index('tid')


    df = pd.read_excel(join(BULGE_DIR, bulge_test_id, SUB_PATH_RESULTS_OVERVIEW, FN_RESULTS_OVERVIEW))

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    df['bta_id'] = bulge_test_id
    df['test_date'] = test_date
    df['memb_id'] = memb_id
    df['memb_cid'] = memb_cassette_id
    df['pre_stretch_nominal'] = memb_pre_stretch_nominal
    df['pre_stretch_measured'] = memb_pre_stretch_measured
    df['thickness_Au'] = thickness_Au

    print(bulge_test_id)

    df = df.join(df_add, on='tid', rsuffix='_add')


    df = df[['bta_id', 'test_date', 'memb_id', 'memb_cid',
             'thickness_um', 'pre_stretch_nominal', 'pre_stretch_measured', 'thickness_Au', 'radius_mm',
             'tid', 'P_last_Pa', 'z_last_um',
             'linear_model_E_MPa', 'nonlinear_model_E_MPa',
             'linear_model_sigma_0_MPa', 'nonlinear_model_sigma_0_MPa',
             'plate_model_E_MPa', 'flexural_rigidity',
             'fit_line_slope_um_per_Pa', 'fit_line_y_intercept_um',
             ]]
    super_df_results.append(df)

    b = 1

super_dfpzs = pd.concat(super_dfpzs)
super_dfpzs.to_excel(join(BULGE_DIR, 'all-bulge-test-analyses-PZ-by-t-by-tid.xlsx'))

super_df_results = pd.concat(super_df_results)
super_df_results.to_excel(join(BULGE_DIR, 'all-bulge-test-analyses-results-overview.xlsx'))
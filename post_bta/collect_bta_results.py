from os.path import join
import os
import numpy as np
import pandas as pd


def get_measured_stretch(memb_casette_id):
    if memb_casette_id == 9:
        return 1.0
    elif memb_casette_id == 13:
        return 1.215
    elif memb_casette_id == 14:
        return 1.126
    elif memb_casette_id == 15:
        return 1.146
    elif memb_casette_id == 16:
        return 1.23
    elif memb_casette_id == 17:
        return 1.252
    elif memb_casette_id == 18:
        return 1.135
    elif memb_casette_id == 19:
        return 1.131
    elif memb_casette_id == 20:
        return 1.17
    elif memb_casette_id == 21:
        return 1.16
    elif memb_casette_id == 22:
        return 1.26
    elif memb_casette_id == 23:
        return 1.0
    elif memb_casette_id == 24:
        return 1.0
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
    print(d)
    bulge_test_id = d
    test_date = d.split('_')[0]
    memb_id = d.split('_')[1]
    bulge_diameter = float(d.split('_')[2].split('mmDia')[0])

    memb_cassette_id = memb_id.split('-')[0].split('C')[1]
    if memb_cassette_id.startswith('X'):
        memb_cassette_id = memb_cassette_id[1:]
    memb_cassette_id = int(memb_cassette_id)
    # memb_cassette_id = int(memb_id.split('-')[0].split('C')[1])

    memb_pre_stretch_nominal = int(memb_id.split('-')[1].split('pT')[0])
    memb_pre_stretch_measured = get_measured_stretch(memb_cassette_id)
    deposition_Au = memb_id.split('-')[2].split('nmAu')[0]
    thickness_Au = eval(deposition_Au)

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
    dfpzs['mmb_id'] = memb_id
    dfpzs['mmb_cid'] = memb_cassette_id
    dfpzs['ps_nom'] = memb_pre_stretch_nominal
    dfpzs['dep_Au'] = deposition_Au
    dfpzs['dia_mm'] = bulge_diameter
    dfpzs['rad_mm'] = bulge_diameter / 2
    dfpzs['ps_meas'] = memb_pre_stretch_measured
    dfpzs['t_Au'] = thickness_Au
    super_dfpzs.append(dfpzs)

    df_add = pd.DataFrame(add_data, columns=['tid', 'dP_Pa', 'dz_um']).set_index('tid')


    df = pd.read_excel(join(BULGE_DIR, bulge_test_id, SUB_PATH_RESULTS_OVERVIEW, FN_RESULTS_OVERVIEW))

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    df['bta_id'] = bulge_test_id
    df['test_date'] = test_date
    df['mmb_id'] = memb_id
    df['mmb_cid'] = memb_cassette_id
    df['ps_nom'] = memb_pre_stretch_nominal
    df['dep_Au'] = deposition_Au
    df['dia_mm'] = df['radius_mm'] * 2
    df['ps_meas'] = memb_pre_stretch_measured
    df['t_Au'] = thickness_Au

    print(bulge_test_id)

    df = df.join(df_add, on='tid', rsuffix='_add')

    df = df.rename(columns={
        'radius_mm': 'rad_mm',
        'thickness_um': 't0_um',
        'pre_stretch': 'ps_bta',
        'thickness_um_post_stretch': 't_ps_um',
        'fit_line_slope_um_per_Pa': 'slope_um_Pa',
        'fit_line_y_intercept_um': 'intercept_um',
        'flexural_rigidity': 'D',
        'linear_model_E_MPa': 'l_E_MPa',
        'linear_model_sigma_0_MPa': 'l_s0_MPa',
        'nonlinear_model_E_MPa': 'nl_E_MPa',
        'nonlinear_model_sigma_0_MPa': 'nl_s0_MPa',
    })

    df['dz/dP'] = df['dz_um'] / df['dP_Pa']
    df['dz/dia'] = (df['dz_um'] * 1e-6) / (df['dia_mm'] * 1e-3)

    df = df[['bta_id', 'ps_nom', 'dep_Au',
             't_Au', 'ps_meas', 'ps_bta', 't_ps_um',
             'dia_mm', 'tid', 'Pmin_Pa', 'Pmax_Pa', 'dP_Pa', 'dz_um', 'dz/dP', 'dz/dia',
             'l_s0_MPa', 'nl_s0_MPa',
             'l_E_MPa', 'nl_E_MPa', 'LB_E_MPa', 'UB_E_MPa',
             'slope_um_Pa', 'intercept_um',
             'D', 'rad_mm', 't0_um', 'plate_model_E_MPa',
             'test_date', 'mmb_id', 'mmb_cid',
             ]]
    super_df_results.append(df)

super_dfpzs = pd.concat(super_dfpzs)
super_dfpzs = super_dfpzs[[
    'bta_id', 'test_date', 'mmb_id', 'mmb_cid', 'ps_nom', 'dep_Au', 'dia_mm', 'rad_mm', 'ps_meas', 't_Au',
    'tid', 't', 'P', 'z',
]]
super_dfpzs.to_excel(join(BULGE_DIR, 'all-bulge-test-analyses-PZ-by-t-by-tid.xlsx'), index=False)

super_df_results = pd.concat(super_df_results)
super_df_results = super_df_results.round({
    't_ps_um': 1,
    'Pmin_Pa': 1,
    'Pmax_Pa': 1,
    'dP_Pa': 1,
    'dz_um': 1,
    'dz/dP': 3,
    'dz/dia': 3,
    'l_s0_MPa': 3,
    'nl_s0_MPa': 3,
    'slope_um_Pa': 3,
    'intercept_um': 2,
})
super_df_results.to_excel(join(BULGE_DIR, 'all-bulge-test-analyses-results-overview.xlsx'), index=False)
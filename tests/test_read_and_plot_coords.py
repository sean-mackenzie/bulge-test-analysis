import pandas as pd
from utils import io, plotting

if __name__ == '__main__':

    FILEPATH = '/Users/mackenzie/Desktop/Bulge Test/BulgeTest_200umSILP_4mmDia/results/' \
               'test-testset-17/test_coords_tILP_4mmDia_cSILPURAN_ccSILPURAN_testset-17.xlsx'
    FRAME_RATE = 16.871

    df = io.read_coords(filepath=FILEPATH, scale_z=1, frame_rate=FRAME_RATE, skip_frame_zero=True)
    plotting.show_z_by_dt(df, savepath=None)
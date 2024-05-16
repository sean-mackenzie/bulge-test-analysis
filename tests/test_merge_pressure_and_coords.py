from os.path import join
import glob
from utils import io, plotting

if __name__ == '__main__':

    BASE_DIR = '/Users/mackenzie/Desktop/Bulge Test/BulgeTest_200umSILP_4mmDia'
    FDIR = join(BASE_DIR, 'results')
    FTYPE = '.xlsx'
    SUBSTRING = 'test_coords_'
    SORT_STRINGS = ['N_testset-', '.xlsx']
    FRAME_RATE = 16.871
    SKIP_FRAME_ZERO = True

    dfs = io.combine_coords(fdir=FDIR, substring=SUBSTRING, sort_strings=SORT_STRINGS,
                            frame_rate=FRAME_RATE, skip_frame_zero=SKIP_FRAME_ZERO)
    dfs.to_excel(join(BASE_DIR, 'combined_coords.xlsx'), index=False)
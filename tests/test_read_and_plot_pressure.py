from utils import io, plotting

if __name__ == '__main__':

    FILEPATH = '/Users/mackenzie/Desktop/Bulge Test/BulgeTest_200umSILP_4mmDia/pressure/test17.txt'
    PX, PY = 'dt', 'P'

    df = io.read_pressure_txt(FILEPATH, cols=None)
    plotting.show_P_by_dt(df)
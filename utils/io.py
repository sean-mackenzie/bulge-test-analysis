from os.path import join
import os
import glob
import pandas as pd


def read_pressure_txt(filepath, cols=None, correction_function=1):
    """
    df = io.read_pressure_txt(filepath, cols=None)
    :param filepath:
    :param cols:
    :return:
    """
    if cols is None:
        cols = ['t_reset', 'P', 'T']
    df = pd.read_csv(filepath, names=cols, skiprows=1)
    df['dt_ms'] = df['t_reset'] - df['t_reset'].iloc[0]
    df['dt'] = df['dt_ms'] / 1000
    df['P'] = df['P'] * correction_function
    return df


def combine_pressure_txt(fdir, ftype, fn_strings, cols, correction_function, fn_startswith=None):
    """
    dfs = io.combine_pressure_txt(fdir, ftype, fn_strings)
    :param fdir:
    :param ftype:
    :param fn_strings:
    :param cols:
    :param correction_function:
    :return:
    """
    files = find_files(fdir=fdir, ftype=ftype, fn_startswith=fn_startswith)
    files, names = sort_files(files, sort_strings=[fn_strings[0], fn_strings[1]])

    t_curr, t_reset_group = 0, 0
    dfs = []
    for f, n in zip(files, names):
        df = read_pressure_txt(filepath=join(fdir, f), cols=cols, correction_function=correction_function)
        # set test id
        df['tid'] = n
        # check against current time since reset
        if df['t_reset'].iloc[0] < t_curr:
            t_reset_group += 1
        # set reset group
        df['group_reset'] = t_reset_group
        # set current time
        t_curr = df['t_reset'].iloc[-1]
        dfs.append(df)
    dfs = pd.concat(dfs)
    return dfs


def read_coords(filepath, scale_z=1, z0=0, flip_z=False, frame_rate=None, skip_frame_zero=False, only_pids=None):
    """
    df = io.read_coords(filepath, frame_rate=None, skip_frame_zero=False)
    :param filepath:
    :param z0:
    :param flip_z:
    :param frame_rate:
    :param skip_frame_zero:
    :return:
    """
    df = pd.read_excel(filepath)

    # three cases for formatting coords.xlsx
    if 'xg' in df.columns:
        df = df[['frame', 'id', 'xg', 'yg', 'z', 'cm']]
        df = df.rename(columns={'xg': 'x', 'yg': 'y'})
    elif 'cm_discrete' in df.columns:
        df = df[['frame', 'id', 'x_sub', 'y_sub', 'z_sub', 'cm_discrete']]
        df = df.rename(columns={'x_sub': 'x', 'y_sub': 'y', 'z_sub': 'z', 'cm_discrete': 'cm'})
    else:
        for col in ['z_true', 'max_sim', 'error']:
            if col in df.columns:
                df = df.drop(columns=col)
    df['z_raw'] = df['z']
    # scale z
    df['z'] = df['z'] * scale_z
    # get only specified pids
    if only_pids is not None:
        df = df[df['id'].isin(only_pids)]
    # skip frame zero
    if skip_frame_zero:
        df = df[df['frame'] > 0]
    # subtract arbitrary z0
    if isinstance(z0, (tuple, list)):
        z0 = df[df['frame'] < z0[1]]['z'].mean()
    df['z'] = df['z'] - z0
    # flip z-axis
    if flip_z:
        df['z'] = df['z'] * -1
    # convert frames into time
    if frame_rate is not None:
        df['dt'] = df['frame'] / frame_rate
    return df


def combine_coords(fdir, substring, sort_strings, scale_z=1, z0=0, flip_z=False,
                   frame_rate=None, skip_frame_zero=False, only_pids=None):
    files = find_subfiles(fdir=fdir, substring=substring)
    files, names = sort_files(files, sort_strings=sort_strings)
    dfs = []
    for f, n in zip(files, names):
        df = read_coords(f, scale_z=scale_z, z0=z0, flip_z=flip_z, frame_rate=frame_rate,
                         skip_frame_zero=skip_frame_zero, only_pids=only_pids)
        # set test id
        df['tid'] = n
        dfs.append(df)
    dfs = pd.concat(dfs)
    return dfs


def find_files(fdir, ftype, fn_startswith=None):
    if fn_startswith is not None:
        files = [f for f in os.listdir(fdir) if f.startswith(fn_startswith) and f.endswith(ftype)]
    else:
        files = [f for f in os.listdir(fdir) if f.endswith(ftype)]
    return files

def find_subfiles(fdir, substring):
    files = [f for f in glob.glob(join(fdir, '**'), recursive=True) if substring in f]
    return files

def sort_files(files, sort_strings):
    files = sorted(files, key=lambda x: float(x.split(sort_strings[0])[-1].split(sort_strings[1])[0]))
    names = [int(f.split(sort_strings[0])[-1].split(sort_strings[1])[0]) for f in files]
    return files, names
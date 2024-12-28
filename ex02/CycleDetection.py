"""Main cycle detection algorithm with auxiliary functions."""
import argparse
import os.path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def detect_cycles(series, drop_zero_docs=True, integer_index=False):
    if len(series) < 4:
        msg = ('The number of elements in the input time series to form one '
               'cycle must be 4 at least.')
        raise ValueError(msg)
    # convert input data to a data frame
    series = series.to_frame(name='values')
    series['id'] = series.index
    series.index = pd.RangeIndex(len(series.index))

    # norm input data
    series['norm'] = series['values'] - series['values'].min()
    maximum = series['norm'].max()
    if maximum == 0:
        msg = 'Detected constant time series.'
        raise ValueError(msg)
    series['norm'] /= maximum

    # find minima and maxima
    min_idx, max_idx = find_peaks_valleys_idx(series['norm'])
    # find start and end times of cycles (potenial start/end at local maxima)
    t_start = soc_find_start(series['norm'], max_idx)
    t_end = soc_find_end(series['norm'], max_idx)

    # search for precycles
    precycles = search_precycle(series['norm'], max_idx, t_start, t_end)

    # cycle detection
    cycles = cycling(precycles)

    # calculate the amplitude of the cycles
    cycles[:, 3] = calc_doc(series['norm'], cycles)

    # write data to DataFrame
    df = pd.DataFrame()
    if integer_index is True:
        # use the integer index as time stamps
        df['t_start'] = cycles[:, 0]
        df['t_end'] = cycles[:, 1]
        df['t_minimum'] = cycles[:, 2]

    else:
        # use original index as time stamps
        df['t_start'] = series.iloc[cycles[:, 0]]['id'].values
        df['t_end'] = series.iloc[cycles[:, 1]]['id'].values
        df['t_minimum'] = series.iloc[cycles[:, 2]]['id'].values

    # write depth of cycle in DataFrame
    df['doc'] = cycles[:, 3]
    # calculate duration
    df['duration'] = df['t_end'] - df['t_start']

    # drop cycles where the amplitude (doc) is zero
    if drop_zero_docs is True:
        df = df.drop(df[df['doc'] == 0].index)

    # reset the index
    df = df.reset_index(drop=True)

    return df


def find_peaks_valleys_idx(series):
    # calculate difference between neighboring values
    diff = np.diff(np.concatenate(([0], series, [0])))
    # calculate sign changes
    asign = np.sign(diff)

    # remove sign changes for zero
    sz = asign == 0
    while sz.any():
        asign[sz] = np.roll(asign, 1)[sz]
        sz = asign == 0
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)

    min_idx = []
    max_idx = []
    start = series[0]

    # iterate over indices of sign changes
    for index in np.where(signchange == 1)[0]:
        if start > 0:
            max_idx += [index - 1]
            start = diff[index]
        else:
            min_idx += [index - 1]
            start = diff[index]

    # no valley found
    if len(min_idx) == 0:
        msg = 'No valleys detected in time series.'
        raise ValueError(msg)

    # no peaks found
    if len(max_idx) == 0:
        msg = 'No peaks detected in time series.'
        raise ValueError(msg)

    # remove first sign change, if at first value of series
    if min_idx[0] <= 0:
        min_idx = min_idx[1:]
    if max_idx[0] <= 0:
        max_idx = max_idx[1:]

    return min_idx, max_idx


def soc_find_start(series, indices):
    t = []

    # number of peaks
    size = len(indices)

    # determine possible starting points for precycles:
    # searches the peaks from the end to the start for a peak at t1 that is
    # higher than the last peak at t2
    # if a peak is found or all preceding peaks have been searched
    # t2 is moved to its predecessor
    for c in range(1, size)[::-1]:
        for d in range(c)[::-1]:
            if series[indices[d]] >= series[indices[c]]:
                t += [indices[d]]
                break
            elif d == 0:
                t += [0]
                break

    t += [indices[-1]]
    return t[::-1]


def soc_find_end(series, indices):
    t = []

    # number of peaks
    size = len(indices)

    # determine possible ending points for precycles:
    # searches the peaks from start to end for a peak at t2 that is
    # higher than the first peak at t1
    # if a peak is found or all preceding peaks have been searched
    # t1 is moved to its successor
    for c in range(size):
        for d in range(c + 1, size):
            if series[indices[d]] >= series[indices[c]]:
                t += [indices[d]]
                break
            elif d == size - 1:
                t += [-1]
                break

    t += [-1]
    return t


def search_precycle(series, indices, t_start, t_end):
    size = len(indices)
    pre_a = np.zeros((size, 4))
    pre_b = np.zeros((size, 4))

    for c in range(size):
        if t_start[c] < indices[c]:
            # start time must be before time of corresponding peak
            # retrieve the values of the series between start time and
            # next peak
            values = series[t_start[c]:indices[c] + 1].tolist()

            # start time
            pre_a[c, 0] = t_start[c]
            # next peak
            pre_a[c, 1] = indices[c]
            # index/time of minimum value
            pre_a[c, 2] = values.index(min(values)) + t_start[c]
            # minimum value of the section
            pre_a[c, 3] = min(values)

        if t_end[c] > indices[c]:
            # end time must be after time of corresponding peak
            # retrieve the values of the series between previous peak and
            # end time
            values = series[indices[c]:t_end[c] + 1].tolist()

            # previous peak
            pre_b[c, 0] = indices[c]
            # end time
            pre_b[c, 1] = t_end[c]
            # index/time of minimum value
            pre_b[c, 2] = values.index(min(values)) + indices[c]
            # minimum value of the section
            pre_b[c, 3] = min(values)

    # concatenate the two arrays
    pre = np.append(pre_a, pre_b, axis=0)
    # remove rows with only zeros (=no precycle found)
    pre = pre[~np.all(pre == 0, axis=1)]

    if len(pre) == 0:
        msg = 'No cycles detected.'
        raise ValueError(msg)
    # remove duplicates
    pre = np.unique(pre, axis=0)
    return pre


def cycling(rows):
    # remove overlaps in the precycles and flag the corresponding rows
    for c in range(rows.shape[0]):
        indices = np.where((rows[c, 2] == rows[:, 2]) &
                           (rows[c, 0] <= rows[:, 0]) &
                           (rows[c, 1] >= rows[:, 1]))[0]
        # choose indices to flag rows
        indices = indices[indices != c]
        rows[indices] = 0

    # remove the flagged rows (rows with zeros only)
    rows = rows[~np.all(rows == 0, axis=1)]

    return rows


def calc_doc(series, rows):
    num = rows.shape[0]
    doc = np.zeros(num)

    for c in range(num):
        doc[c] = min(series[rows[c, 0]], series[rows[c, 1]]) - rows[c, 3]
    return doc


def process2dpts(filepath=None):
    # Load 2D joint data from CSV file
    df = pd.read_csv(args.i, sep=',')
    _r = df.shape[0]
    _c = df.shape[1]
    # print(_r, _c)
    _dists = pd.DataFrame()
    _angls = pd.DataFrame()

    for _ci in range(1, _c, 2):
        # Convert 2D time-series data to 1D time series data
        _df = df.iloc[:, _ci:_ci + 2]
        # Get joint ID
        _id = _df.columns.tolist()[0].split(' ')[0]

        _df = _df.interpolate(method='linear')
        _df.columns = ["x", "y"]

        # Calculate the distance from origin (0,0)
        _cd = detect_cycles(np.sqrt(_df['x'] ** 2 + _df['y'] ** 2))
        _cd = _cd.dropna()
        # _distances.columns = ["JointID", "t_start", "t_end", "t_minimum", "doc", "duration"]
        _cd["JointID"] = _id
        _dists = pd.concat([_dists, _cd], ignore_index=False)

        # Calculate the angle with respect to the x-axis for each point
        _ca = detect_cycles(np.arctan2(_df['y'], _df['x']))
        _ca = _ca.dropna()
        _ca["JointID"] = _id
        _angls = pd.concat([_angls, _cd], ignore_index=False)

    # Perform agglomerative clustering
    kmeans = KMeans(n_clusters=3)
    dc = kmeans.fit_predict(_dists[["t_start", "t_end", "t_minimum", "doc", "duration", "JointID"]])
    ac = kmeans.fit_predict(_angls[["t_start", "t_end", "t_minimum", "doc", "duration", "JointID"]])

    # Perform majority voting over clusters form dc and ac


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform cycle detection over time-series data')
    parser.add_argument('-i', type=str, help='Input csv file containing time-series data', required=True)
    args = parser.parse_args()

    if os.path.exists(args.i):
        process2dpts(filepath=args.i)

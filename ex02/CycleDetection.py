import argparse
import os.path
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from collections import defaultdict

def detect_cycles(series, drop_zero_docs=True, integer_index=False):
    """Original cycle detection function remains unchanged"""
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

    try:
        # find minima and maxima
        min_idx, max_idx = find_peaks_valleys_idx(series['norm'])
        # find start and end times of cycles
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
            df['t_start'] = cycles[:, 0]
            df['t_end'] = cycles[:, 1]
            df['t_minimum'] = cycles[:, 2]
        else:
            df['t_start'] = series.iloc[cycles[:, 0]]['id'].values
            df['t_end'] = series.iloc[cycles[:, 1]]['id'].values
            df['t_minimum'] = series.iloc[cycles[:, 2]]['id'].values

        df['doc'] = cycles[:, 3]
        df['duration'] = df['t_end'] - df['t_start']

        if drop_zero_docs is True:
            df = df.drop(df[df['doc'] == 0].index)

        df = df.reset_index(drop=True)
        return df
    
    except Exception as e:
        print(f"Warning: Cycle detection failed with error: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error


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


def analyze_joint_cycles(filepath):
    """Analyze cycles in joint trajectories"""
    # Load joint trajectory data
    df = pd.read_csv(filepath)
    
    # Initialize results dictionary
    results = defaultdict(dict)
    
    # Process each joint
    joint_names = [col.split('_')[0] for col in df.columns if col.endswith('_x')]
    
    for joint in joint_names:
        print(f"Analyzing cycles for {joint}...")
        
        # Calculate 3D trajectory
        x = df[f'{joint}_x']
        y = df[f'{joint}_y']
        z = df[f'{joint}_z']
        
        # Calculate various movement metrics
        try:
            # 3D distance from origin
            distance_3d = np.sqrt(x**2 + y**2 + z**2)
            cycles_3d = detect_cycles(distance_3d)
            
            # Planar projections
            distance_xy = np.sqrt(x**2 + y**2)
            cycles_xy = detect_cycles(distance_xy)
            
            distance_yz = np.sqrt(y**2 + z**2)
            cycles_yz = detect_cycles(distance_yz)
            
            distance_xz = np.sqrt(x**2 + z**2)
            cycles_xz = detect_cycles(distance_xz)
            
            # Store results
            results[joint] = {
                "cycles_3d": len(cycles_3d),
                "cycles_xy": len(cycles_xy),
                "cycles_yz": len(cycles_yz),
                "cycles_xz": len(cycles_xz),
                "details": {
                    "avg_duration_3d": float(cycles_3d['duration'].mean()) if not cycles_3d.empty else 0,
                    "avg_amplitude_3d": float(cycles_3d['doc'].mean()) if not cycles_3d.empty else 0
                }
            }
            
        except Exception as e:
            print(f"Warning: Failed to analyze {joint}: {str(e)}")
            results[joint] = {
                "error": str(e),
                "cycles_3d": 0,
                "cycles_xy": 0,
                "cycles_yz": 0,
                "cycles_xz": 0,
                "details": {
                    "avg_duration_3d": 0,
                    "avg_amplitude_3d": 0
                }
            }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Detect cycles in joint motion trajectories')
    parser.add_argument('-i', '--input', type=str, required=True,
                       help='Input CSV file containing joint trajectories')
    parser.add_argument('-o', '--output', type=str, default='cycles.json',
                       help='Output JSON file for cycle analysis results')
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    # Analyze cycles
    print(f"Analyzing cycles in: {args.input}")
    results = analyze_joint_cycles(args.input)
    
    # Save results to JSON
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")
    
    # Print summary
    print("\nSummary of detected cycles:")
    for joint, data in results.items():
        print(f"\n{joint}:")
        print(f"  3D cycles: {data['cycles_3d']}")
        print(f"  XY plane cycles: {data['cycles_xy']}")
        print(f"  YZ plane cycles: {data['cycles_yz']}")
        print(f"  XZ plane cycles: {data['cycles_xz']}")

if __name__ == "__main__":
    main()

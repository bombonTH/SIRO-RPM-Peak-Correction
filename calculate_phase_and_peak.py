import math

import numpy as np
from peakdetect import peakdetect


def calculate_peak_and_phase(data):
    amplitude = data[:, 0].astype(float)
    timestamp = data[:, 2].astype(int)
    peaks = peakdetect(amplitude, timestamp, lookahead=15, delta=0.1)
    high_peaks = np.array(peaks[0])
    low_peaks = np.array(peaks[1])
    z_marks = low_peaks[:, 0].astype(int)
    p_marks = high_peaks[:, 0].astype(int)
    marks = ["P" if time in p_marks else "Z" if time in z_marks else "" for time in timestamp]
    data[:, 5] = marks
    phases = np.zeros_like(timestamp).astype(float)

    last_mark_time = min(z_marks[0], p_marks[0])
    next_mark_time = 0
    next_z_index = 0
    next_p_index = 0
    calculating = False
    for index, value in enumerate(marks):
        current_phase = 0
        if value == "Z":
            current_phase = 0
            calculating = True
            next_z_index += 1
            last_mark_time = timestamp[index]
            try:
                next_mark_time = p_marks[next_p_index]
            except IndexError:
                calculating = False
        elif value == "P":
            current_phase = math.pi
            calculating = True
            next_p_index += 1
            last_mark_time = timestamp[index]
            try:
                next_mark_time = z_marks[next_z_index]
            except IndexError:
                calculating = False
        elif calculating:
            last_phase = phases[index - 1]
            time_elapse = timestamp[index] - timestamp[index - 1]
            time_between_marks = next_mark_time - last_mark_time
            current_phase = last_phase + math.pi * time_elapse / time_between_marks
        phases[index] = current_phase
    data[:, 1] = ["%.4f" % phase for phase in phases]
    return data

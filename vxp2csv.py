import argparse
import csv
import os

import math
import matplotlib.pyplot as plt
from scipy.signal import sosfiltfilt, butter, filtfilt, sosfilt
import scipy.stats as st
import numpy as np
from peakdetect import peakdetect
import scipy.fftpack


from calculate_phase_and_peak import calculate_peak_and_phase


def read_all_patients(input_folder):
    return [name for name in os.listdir(input_folder) if name.endswith(".vxp")]


def get_header(input_folder, filename):
    output = dict()
    with open(input_folder + filename, 'r') as fle:
        for line in fle:
            if '[Header]' in line:
                for header in fle:
                    if header.startswith('[Data]'):
                        return output
                    header, value = header.split('=')
                    output[header] = value.rstrip()
    return output


def get_data(input_folder, filename):
    output = []
    with open(input_folder + filename, 'r') as fle:
        for line in fle:
            if line.startswith('[Data'):
                for data in fle:
                    output.append(data.strip().split(","))
    return output


def write_data(output_folder, filename, headers, data):
    os.makedirs(os.path.dirname(output_folder), exist_ok=True)
    filename = filename + '.csv'
    with open(output_folder + filename, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(headers["Data_layout"].split(","))
        writer.writerows(data)
    return filename


def plot(amplitude, ttlin_mask, z_mask, p_mask, ax):
    ttlin_index = np.where(ttlin_mask)
    first_ttlin = ttlin_index[0][0]
    last_ttlin = ttlin_index[0][-1]
    total_time = len(amplitude) * 40
    start_time = -first_ttlin * 40
    stop_time = total_time + start_time
    timestamp = np.array(range(start_time, stop_time, 40))
    ax.plot(timestamp, amplitude)
    ax.axvspan(start_time, 0, color='r', alpha=0.3)
    ax.axvspan(timestamp[last_ttlin+1], stop_time, color='r', alpha=0.3)
    z_annotate = np.stack((timestamp[z_mask] - 1200, amplitude[z_mask] - 0.02), axis=-1)
    p_annotate = np.stack((timestamp[p_mask] - 1200, amplitude[p_mask] + 0.01), axis=-1)
    for z in z_annotate:
        ax.annotate('Z', z)
    for p in p_annotate:
        ax.annotate('P', p)
    ax.invert_yaxis()
    ax.set_ylabel('x (cm)')
    ax.set_xlabel('t (ms)')
    
            
def plotMEIE(info, ax, color='r'):
    ax.axhline(info['MEE'], color=color)
    ax.axhline(info['MEI'], color=color)


def calculate_info(amplitude, ttlin_mask, z_mask, p_mask):
    z_and_ttlin = np.logical_and(ttlin_mask, z_mask)
    p_and_ttlin = np.logical_and(ttlin_mask, p_mask)

    MEE = np.mean(amplitude[z_and_ttlin]) 
    STDEE = np.std(amplitude[z_and_ttlin])
    MUD = np.min(amplitude[p_and_ttlin])  
    MDD = np.max(amplitude[p_and_ttlin])  
    MEI = np.mean(amplitude[p_and_ttlin])    
    STDEI = np.std(amplitude[p_and_ttlin])
    
    z_index = np.where(z_and_ttlin)[0]
    time_btw_z = np.ediff1d(z_index)
    period_mean = np.mean(time_btw_z)
    period_std = np.std(time_btw_z)
    return {'MEE': MEE, 'STDEE': STDEE, 'MEI':MEI, 'STDEI':STDEI,'MDD': MDD,'MUD': MUD, 'PERIOD_MEAN': period_mean, 'PERIOD_STD':period_std}
    
    
def filter_signal(amplitude, sample_rate, filename):
    print('Applying filter')
    median = np.median(amplitude)
    amplitude = amplitude - median
    sos = butter(N=50, Wn=0.75, btype='low', output='sos', fs=sample_rate)
    sos = butter(N=50, Wn=(0.02, 1.00) , btype='band', output='sos', fs=sample_rate)
    amplitude = sosfiltfilt(sos, amplitude)
    return amplitude, filename
    

def detect_peak(amplitude):
    index = range(len(amplitude))
    peaks = peakdetect(amplitude, index , lookahead=15, delta=0.01)
    high_peaks = np.array(peaks[0])
    low_peaks = np.array(peaks[1])
    z_marks = low_peaks[:, 0].astype(int)
    p_marks = high_peaks[:, 0].astype(int)
    z_mask = [i in z_marks for i in index]
    p_mask = [i in p_marks for i in index]
    return z_mask, p_mask
    

def calculate_phase(z_mask, p_mask):
    z_mask = np.copy(z_mask)
    z_mask[0] = True
    z_index = np.where(z_mask)[0]
    time_btw_z = np.ediff1d(z_index)
    period_mean = np.mean(time_btw_z)
    period_std = np.std(time_btw_z)
    last_period = z_mask.shape[0] - z_index[-1]
    time_btw_z = np.append(time_btw_z, [len(z_mask) - z_index[-1]])
    phase = np.zeros(len(z_mask))
    j = -1
    for i in range(len(z_mask)):
        if z_mask[i]:
            phase[i] = 0.
            j += 1
        else:
            phase[i] = phase[i-1] + (math.pi * 2) / time_btw_z[j]
    return(phase)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--all", help="Convert all files", action="store_true")
    parser.add_argument("-f", "--file", help="File to be converted", type=str)
    parser.add_argument("-n", "--no-phase-correction", help="Do not correct peaks and phase", action="store_true")
    parser.add_argument("-p", "--plot", help="Plot the result", action="store_true")
    parser.add_argument("-l", "--lowpass", help="Apply lowpass filter", action="store_true")
    args = parser.parse_args()
    args.all = args.file is None
    input_folder = './data/in/'
    output_folder = './data/out/'

    if args.all:
        print(f'Converting all .vxp file in {input_folder}')
    if args.file:
        print(f'Converting {args.file}')
    print(f'Phase correction: {"Disabled" if args.no_phase_correction else "Enabled"}')
    print(f'Plotting: {"Enabled" if args.plot else "Disabled"}')
    print('###########################################################')

    if args.all:
        files = read_all_patients(input_folder)
    else:
        files = [args.file]

    for filename in files:
        print(f'Reading {filename}')
        headers = get_header(input_folder, filename)
        data = get_data(input_folder, filename)
        sample_rate = float(headers["Samples_per_second"])
        scale_factor = 10 / float(headers["Scale_factor"])
        print(f'File: {filename} - Date: {headers["Date"]} - Sampling Rate: {sample_rate}')
        raw_data = np.array(data)
        raw_amplitude = raw_data[:,0].astype(float) / scale_factor
        median = np.median(raw_amplitude)
        raw_amplitude = raw_amplitude - median  
        ttlin_mask = np.array([ttlin == 0 for ttlin in raw_data[:, 4].astype(int)])
        raw_z_mask = np.array([mark == "Z" for mark in raw_data[:, 5]])
        raw_p_mask = np.array([mark == "P" for mark in raw_data[:, 5]])       
        raw_info = calculate_info(raw_amplitude, ttlin_mask, raw_z_mask, raw_p_mask)
        raw_phase = raw_data[:,1].astype(float)
        raw_marks = raw_data[:,5]
        timestamp = raw_data[:,2].astype(int)
        
        new_amplitude = np.copy(raw_amplitude)
        new_z_mask = np.copy(raw_z_mask)
        new_p_mask = np.copy(raw_p_mask)
        new_info = np.copy(raw_info)
        new_phase = np.copy(raw_phase)   
        new_marks = np.copy(raw_marks)
        
        if args.lowpass:
            new_amplitude, filename = filter_signal(new_amplitude, sample_rate, filename)
            filename = filename.split(".")[0]
            filename += "_FILTERED"
        
        if not args.no_phase_correction:
            new_z_mask, new_p_mask = detect_peak(new_amplitude)
            new_info = calculate_info(new_amplitude, ttlin_mask, new_z_mask, new_p_mask)
            new_phase = calculate_phase(new_z_mask, new_p_mask)
            new_marks = ["Z" if z_mask else "" for z_mask in new_z_mask]
            new_marks = np.core.defchararray.add(new_marks, ["P" if p_mask else "" for p_mask in new_p_mask])
            filename = filename.split(".")[0]
            filename += "_CORRECTED"

        if args.plot:
            fig, (ax11, ax21) = plt.subplots(nrows=2, ncols=1, sharey=False)
            fig.suptitle(t=f'File: {filename} - Date: {headers["Date"]} - Sampling Rate: {headers["Samples_per_second"]}')
    
            ax11.set(title=f'Inhale: {raw_info["MEE"]:.3f}$\pm${raw_info["STDEE"]:.3f} Exhale: {raw_info["MEI"]:.3f}$\pm${raw_info["STDEI"]:.3f} MUD-MDD: {raw_info["MUD"]-raw_info["MDD"]:.3f}')
            plot(raw_amplitude, ttlin_mask, raw_z_mask, raw_p_mask, ax11)
            plotMEIE(raw_info, ax11, 'g')
            
            ax21.set(title=f'Inhale: {new_info["MEE"]:.3f}$\pm${new_info["STDEE"]:.3f} Exhale: {new_info["MEI"]:.3f}$\pm${new_info["STDEI"]:.3f} MUD-MDD: {new_info["MUD"]-new_info["MDD"]:.3f} PERIOD: {new_info["PERIOD_MEAN"] / int(headers["Samples_per_second"]):.3f}s $\pm${new_info["PERIOD_STD"] / int(headers["Samples_per_second"]):.3f}s')
            plot(new_amplitude, ttlin_mask, new_z_mask, new_p_mask, ax21)
            plotMEIE(new_info, ax21, 'b')
            
            plt.show()
        
        new_data = np.stack((new_amplitude, new_phase, raw_data[:, 2], raw_data[:, 3], raw_data[:, 4], new_marks, raw_data[:, 6]),axis=-1)
        filename = write_data(output_folder, filename, headers, new_data)
        print(f'Output written to {output_folder}{filename}')
        print('###########################################################')


if __name__ == "__main__":
    main()

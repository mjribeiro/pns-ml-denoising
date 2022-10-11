from audioop import minmax
from xml.etree.ElementInclude import LimitedRecursiveIncludeError
import numpy as np
import plotly.graph_objects as go

from scipy import signal


def bandpass_filter(data, freqs=[100, 10e3], fs=100e3, order=4):
    """
    Apply low-pass filter using scipy's signal functions.
    """
    b, a = signal.butter(order, freqs, fs=fs, btype='bandpass')
    data_filt = signal.lfilter(b, a, data)

    return data_filt


def extract_window(data, fs=100e3, start=0, win_length=0.008):
    """
    Takes a 1D array from a recording and returns 20s window
    """
    end = start + (win_length * fs)

    return data[int(start):int(end)]


def generate_dataset(data, fs=100e3, num_channels=9, win_length=0.008):
    """
    Take multiple channel data and return list of 20s windows
    """
    all_columns = data.columns

    num_windows = np.floor(len(data["Channel 1"]) / (win_length*fs))
    vagus_dataset = np.zeros((int(num_windows * num_channels), int(win_length*fs)))

    store_index = 0
    for column in all_columns[1:]:
        channel_data = data[column]
        channel_data = minmax_scaling(remove_artefacts(channel_data))

        for search_index in range(0, len(channel_data), int(win_length*fs)):
            extracted_window = extract_window(channel_data, fs=fs, start=search_index, win_length=win_length)

            if len(extracted_window) < (win_length*fs):
                break
            else:
                vagus_dataset[store_index] = extracted_window

            store_index += 1

    return vagus_dataset

 
def minmax_scaling(data):
    """
    Scales data to [-1, 1]
    """
    return 2 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1

def remove_artefacts(data):
    """
    Remove 15 kHz noise in data and replace with mean of recording.
    """
    # First correct baseline
    orig_mean = np.mean(data)
    data -= orig_mean

    # Then correct artefacts
    channel_mean = np.mean(data)
    limit = 3 * np.std(data)

    data[data > limit] = channel_mean
    data[data < -limit] = channel_mean

    return data


def plot_all_channels(data, filt=False, fs=100e3, lims=np.asarray([0.649, 0.656])):
    """
    Plot all channels in a pig vagus recording (stacked plot)
    """
    # Initialise figure and appearance
    fig = go.Figure()
    fig.update_layout(
        title="Pig vagus spontaneous activity: all channels",
        xaxis_title="Time (s)",
        yaxis_title="Voltage (V)"
    )

    all_columns = data.columns
    offset = 0
    lims = (lims * fs).astype(int)

    for column in all_columns[1:]:
        y = data[column]
        y = remove_artefacts(y)

        if filt:
            y = bandpass_filter(y, freqs=[100, 10e3], fs=fs, order=4)

        if len(lims) > 0:
            y = y[lims[0]:lims[1]]
            x = data["Time"][lims[0]:lims[1]]
        
        print(f"Plotting {column}...")
        fig.add_trace(go.Scatter(x=x, y=y+offset, mode='lines', name=column))
        
        offset += 10  # Max amplitude, peak-to-peak

    fig.show()

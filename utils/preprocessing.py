import numpy as np
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from scipy import signal


def bandpass_filter(data, freqs=[100, 5e3], fs=100e3, order=4):
    """
    Apply low-pass filter using scipy's signal functions.
    """
    b, a = signal.butter(order, freqs, fs=fs, btype='bandpass')
    data_filt = signal.filtfilt(b, a, data, axis=0)

    return data_filt


def extract_window(data, fs=100e3, start=0, win_length=0.008):
    """
    Takes a 1D array from a recording and returns 20s window
    """
    end = start + (win_length * fs)

    return data[int(start):int(end)]


def generate_bp_windows(bp_data, fs=100e3, win_length=0.008):
    """
    Split blood pressure data into list of windows of data
    """
    num_windows = int(np.floor(len(bp_data) / (win_length*fs)))
    vagus_bp_data = np.zeros((num_windows, int(win_length*fs)))

    store_index = 0
    for search_index in range(0, len(bp_data), int(win_length*fs)):
        extracted_bp_window =  extract_window(bp_data, fs=fs, start=search_index, win_length=win_length)

        if len(extracted_bp_window) < (int(win_length*fs)):
                break

        vagus_bp_data[store_index, :] = extracted_bp_window
        store_index += 1

    return vagus_bp_data


def generate_datasets(data, bp_data, fs=100e3, num_channels=9, win_length=0.008):
    """
    Take multiple channel data and return list of windows of data
    """
    all_columns = data.columns

    num_windows = int(np.floor(len(data["Channel 1"]) / (win_length*fs)))

    print("Extracting blood pressure windows...")
    vagus_bp_data = generate_bp_windows(bp_data=bp_data,
                                        fs=fs,
                                        win_length=win_length)
    
    print("Extracting ENG windows...")
    vagus_data_raw       = np.zeros((num_windows, num_channels, int(win_length*fs)))
    vagus_data_bp_wide   = np.zeros((num_windows, num_channels, int(win_length*fs)))
    vagus_data_bp_narrow = np.zeros((num_windows, num_channels, int(win_length*fs)))


    for column, ch in zip(all_columns[1:], range(len(all_columns[1:]))):
        store_index = 0
        channel_data = data[column]

        # Standardise mean and standard dev
        scaler = StandardScaler()

        channel_data = scaler.fit_transform(np.asarray(channel_data).reshape(-1, 1))

        # Remove artefacts and rescale to [-1, 1]
        channel_data = minmax_scaling(remove_artefacts(channel_data))
        channel_data_wide_filt = bandpass_filter(channel_data, freqs=[50, 49.9e3])
        channel_data_narrow_filt = bandpass_filter(channel_data, freqs=[100, 10e3])

        for search_index in range(0, len(channel_data), int(win_length*fs)):

            extracted_win_raw         = extract_window(channel_data, fs=fs, start=search_index, win_length=win_length)
            extracted_win_wide_filt   = extract_window(channel_data_wide_filt, fs=fs, start=search_index, win_length=win_length)
            extracted_win_narrow_filt = extract_window(channel_data_narrow_filt, fs=fs, start=search_index, win_length=win_length)

            if len(extracted_win_raw) < (int(win_length*fs)):
                break
            
            vagus_data_raw[store_index, ch, :]       = extracted_win_raw.flatten()
            vagus_data_bp_wide[store_index, ch, :]   = extracted_win_wide_filt.flatten()
            vagus_data_bp_narrow[store_index, ch, :] = extracted_win_narrow_filt.flatten()

            store_index += 1

    print("Combining blood pressure and ENG data into one dataset...")
    # TODO: Find way of appending BP data to existing numpy array (new dimension?)


    return vagus_data_raw, vagus_data_bp_wide, vagus_data_bp_narrow

 
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
        yaxis_title="Voltage (V)",
        font=dict(
            size=32,
        ),
        plot_bgcolor='white'
    )
    fig.update_xaxes(
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
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

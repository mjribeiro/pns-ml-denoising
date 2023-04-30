import numpy as np
import plotly.graph_objects as go
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import signal
from typing import Tuple


def bandpass_filter(data, freqs=[100, 5e3], fs=100e3, order=4):
    """
    Apply low-pass filter using scipy's signal functions.
    """
    b, a = signal.butter(order, freqs, fs=fs, btype='bandpass')
    data_filt = signal.filtfilt(b, a, data, axis=0)

    return data_filt


def calculate_cross_correlation(signal1, signal2):
    """
    Calculate the cross correlation between two signals
    """
    return np.corrcoef(signal1, signal2)


def drop_data(data, drop_start, win_len=1024) -> np.ndarray:
    keep_data = data[drop_start:, :, :]
    window_count = len(keep_data) // win_len
    drop_end = len(keep_data) - (window_count * win_len)
    keep_data = keep_data[:-drop_end, :, :]

    return keep_data, window_count


def make_windows(data, window_count, num_channels, win_len):
    transposed_data = data.transpose(2, 0, 1)
    windowed_data = transposed_data.reshape((num_channels, window_count, win_len, 2))

    transposed_win_data = windowed_data.transpose(1, 0, 2, 3)

    return transposed_win_data



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
    fs = int(fs)
    all_columns = data.columns

    # NEW
    # Split, preprocess, window

    # Store all channel data not as windows
    noisy_inputs  = np.zeros((len(data["Channel 1"]), 2, num_channels))
    noisy_targets = np.zeros((len(data["Channel 1"]), 2, num_channels))

    for column, ch in zip(all_columns[1:], range(len(all_columns[1:]))):
        channel_data = data[column].to_numpy()
        channel_data_narrow_filt = bandpass_filter(channel_data, freqs=[250, 10e3])

        noisy_inputs[:, :, ch]  = np.asarray([channel_data, bp_data]).T
        noisy_targets[:, :, ch] = np.asarray([channel_data_narrow_filt, bp_data]).T
    
    # Split into train/validate/test
    n2n_X_train, n2n_X_test, n2n_y_train, n2n_y_test = train_test_split(noisy_inputs, noisy_targets, test_size=0.2, shuffle=False)
    n2n_X_train, n2n_X_val, n2n_y_train, n2n_y_val = train_test_split(n2n_X_train, n2n_y_train, test_size=0.1, shuffle=False)

    # Preprocess data - normalise, remove artefacts, minmax scaling
    for ch in range(num_channels):
        # Standardise mean and standard dev
        scaler = StandardScaler()

        n2n_X_train[:, 0, ch] = remove_artefacts(n2n_X_train[:, 0, ch], threshold=2)
        n2n_y_train[:, 0, ch] = remove_artefacts(n2n_y_train[:, 0, ch], threshold=2)

        n2n_X_test[:, 0, ch]  = remove_artefacts(n2n_X_test[:, 0, ch], threshold=2)
        n2n_y_test[:, 0, ch]  = remove_artefacts(n2n_y_test[:, 0, ch], threshold=2)

        n2n_X_val[:, 0, ch]   = remove_artefacts(n2n_X_val[:, 0, ch], threshold=2)
        n2n_y_val[:, 0, ch]   = remove_artefacts(n2n_y_val[:, 0, ch], threshold=2)
        
        n2n_X_train[:, 0, ch] = scaler.fit_transform(np.asarray(n2n_X_train[:, 0, ch]).reshape(-1, 1)).flatten()
        n2n_y_train[:, 0, ch] = scaler.fit_transform(np.asarray(n2n_y_train[:, 0, ch]).reshape(-1, 1)).flatten()

        n2n_X_test[:, 0, ch]  = scaler.fit_transform(np.asarray(n2n_X_test[:, 0, ch]).reshape(-1, 1)).flatten()
        n2n_y_test[:, 0, ch]  = scaler.fit_transform(np.asarray(n2n_y_test[:, 0, ch]).reshape(-1, 1)).flatten()

        n2n_X_val[:, 0, ch]   = scaler.fit_transform(np.asarray(n2n_X_val[:, 0, ch]).reshape(-1, 1)).flatten()
        n2n_y_val[:, 0, ch]   = scaler.fit_transform(np.asarray(n2n_y_val[:, 0, ch]).reshape(-1, 1)).flatten()

        # Remove artefacts and rescale to [-1, 1]
        n2n_X_train[:, 0, ch] = minmax_scaling(n2n_X_train[:, 0, ch])
        n2n_y_train[:, 0, ch] = minmax_scaling(n2n_y_train[:, 0, ch])

        n2n_X_test[:, 0, ch]  = minmax_scaling(n2n_X_test[:, 0, ch])
        n2n_y_test[:, 0, ch]  = minmax_scaling(n2n_y_test[:, 0, ch])

        n2n_X_val[:, 0, ch]   = minmax_scaling(n2n_X_val[:, 0, ch])
        n2n_y_val[:, 0, ch]   = minmax_scaling(n2n_y_val[:, 0, ch])
    
    win_len = int(win_length*fs)

    # Get windows of data for validation and test, omitting any data that wouldn't fit
    n2n_X_test, window_count_X = drop_data(n2n_X_test, 0, win_len)
    n2n_X_test = make_windows(n2n_X_test, window_count_X, num_channels, win_len)
    n2n_y_test, window_count_y = drop_data(n2n_y_test, 0, win_len)
    n2n_y_test = make_windows(n2n_y_test, window_count_y, num_channels, win_len)
    assert window_count_X == window_count_y
    n2n_X_val, window_count_X = drop_data(n2n_X_val, 0, win_len)
    n2n_X_val = make_windows(n2n_X_val, window_count_X, num_channels, win_len)
    n2n_y_val, window_count_y = drop_data(n2n_y_val, 0, win_len)
    n2n_y_val = make_windows(n2n_y_val, window_count_y, num_channels, win_len)
    assert window_count_X == window_count_y

    # Get windows and augment training data
    n2n_X_train, window_count_X = drop_data(n2n_X_train, 0, win_len)
    n2n_X_train_final = make_windows(n2n_X_train, window_count_X, num_channels, win_len)
    n2n_y_train, window_count_y = drop_data(n2n_y_train, 0, win_len)
    n2n_y_train_final = make_windows(n2n_y_train, window_count_y, num_channels, win_len)
    assert window_count_X == window_count_y

    num_augmentation = 2 ** 1
    for i in range(win_len // num_augmentation, win_len, win_len // num_augmentation):
        n2n_X_train_tmp, window_count_X = drop_data(n2n_X_train, i, win_len)
        n2n_X_train_tmp = make_windows(n2n_X_train_tmp, window_count_X, num_channels, win_len)
        n2n_y_train_tmp, window_count_y = drop_data(n2n_y_train, i, win_len)
        n2n_y_train_tmp = make_windows(n2n_y_train_tmp, window_count_y, num_channels, win_len)
        assert window_count_X == window_count_y

        n2n_X_train_final = np.concatenate((n2n_X_train_final, n2n_X_train_tmp), axis=0)
        n2n_y_train_final = np.concatenate((n2n_y_train_final, n2n_y_train_tmp), axis=0)

    n2n_X_train = n2n_X_train_final
    n2n_y_train = n2n_y_train_final

    # # Create new input and target variables with point swaps (Data augmentation following Calvarons 2021 paper)
    # noisy_inputs_arr  = np.zeros((len(n2n_X_train) * 2, num_channels, int(win_length*fs), 2))
    # noisy_targets_arr = np.zeros((len(n2n_y_train) * 2, num_channels, int(win_length*fs), 2))

    # # Store original inputs/targets
    # noisy_inputs_arr[:len(n2n_X_train)] = n2n_X_train
    # noisy_targets_arr[:len(n2n_y_train)] = n2n_y_train

    # for input, target, idx in zip(n2n_X_train, n2n_y_train, range(len(n2n_X_train), len(n2n_y_train) * 2)):
    #     # Pick arbitrary index for points to swap
    #     rand_idx = random.randrange(len(input[ch, :, 0]))
        
    #     for ch in range(num_channels):
    #         # Swap at specific index
    #         input[ch, rand_idx, 0], target[ch, rand_idx, 0] = target[ch, rand_idx, 0], input[ch, rand_idx, 0]

    #         # Store in relevant arrays
    #         noisy_inputs_arr[idx, ch, :, :] = input[ch, rand_idx]
    #         noisy_targets_arr[idx, ch, :, :] = target[ch, rand_idx]

    # print("Finished preparing data for Noise2Noise model...")
    # n2n_X_train = noisy_inputs_arr
    # n2n_y_train = noisy_targets_arr

    # return vagus_data_raw, vagus_data_filt_wide, vagus_data_filt_narrow
    return n2n_X_train, n2n_X_val, n2n_X_test, n2n_y_train, n2n_y_val, n2n_y_test

 
def minmax_scaling(data):
    """
    Scales data to [-1, 1]
    """
    return 2 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1


def prepare_n2n_data(noisy_inputs, filtered_data, bp_data, fs=100e3, num_chs=9, data_len=1024):
    """
    Take noisy and filtered data and prepare to feed into Noise2Noise model.
    Needs independent noisy pairs to begin with, then more can be generated
    """
    print("Started preparing data for Noise2Noise model...")
    
    # Add GWN to filtered data
    # noisy_inputs_new = np.zeros_like(noisy_inputs)
    noisy_targets = np.zeros_like(filtered_data)

    for data_idx in range(len(filtered_data)):
        for ch in range(num_chs):
            # data_noisy = noisy_inputs[data_idx, ch, :, 0]
            data_filt = filtered_data[data_idx, ch, :, 0]

            # Trying making inputs noisier
            # noisy_inputs_new[data_idx, ch, :, 0] = data_noisy + np.random.normal(0, 0.2, len(data_noisy))

            # Not adding noise to filtered data
            noisy_targets[data_idx, ch, :, 0] = data_filt

    # Create hold-out test set
    # n2n_X_train, n2n_X_test, n2n_y_train, n2n_y_test = train_test_split(noisy_inputs, noisy_targets, test_size=0.2, shuffle=False)
    # n2n_X_train, n2n_X_val, n2n_y_train, n2n_y_val = train_test_split(n2n_X_train, n2n_y_train, test_size=0.1, shuffle=False)

    # Normalise each set separately (here) to avoid data leakage?
    # scaler = StandardScaler()

    # for ch in range(num_chs):
    #     n2n_X_train[:, ch, :, 0] = scaler.fit_transform(n2n_X_train[:, ch, :, 0])
    #     n2n_y_train[:, ch, :, 0] = scaler.fit_transform(n2n_y_train[:, ch, :, 0])

    #     n2n_X_val[:, ch, :, 0] = scaler.fit_transform(n2n_X_val[:, ch, :, 0])
    #     n2n_y_val[:, ch, :, 0] = scaler.fit_transform(n2n_y_val[:, ch, :, 0])

    #     n2n_X_test[:, ch, :, 0] = scaler.fit_transform(n2n_X_test[:, ch, :, 0])
    #     n2n_y_test[:, ch, :, 0] = scaler.fit_transform(n2n_y_test[:, ch, :, 0])

        # Remove artefacts and rescale to [-1, 1]
        # channel_data = minmax_scaling(remove_artefacts(channel_data, threshold=2))

    # channel_data = scaler.fit_transform(np.asarray(channel_data).reshape(-1, 1))

    # # Remove spikes in train set, but not test set
    # for window in range(n2n_X_train.shape[0]):
    #     for ch in range(n2n_X_train.shape[1]):
    #         n2n_X_train[window, ch, :, 0] = remove_artefacts(n2n_X_train[window, ch, :, 0], threshold=2)
    
    return n2n_X_train, n2n_X_val, n2n_X_test, n2n_y_train, n2n_y_val, n2n_y_test


def remove_artefacts(data, threshold=2):
    """
    Remove 15 kHz noise in data and replace with mean of recording.
    """
    # First correct baseline
    orig_mean = np.mean(data)
    data -= orig_mean

    # Then correct artefacts
    # channel_mean = np.mean(data)
    channel_median = np.median(data)
    limit = threshold * np.std(data)

    # data[data > limit] = channel_mean
    # data[data < -limit] = channel_mean
    data[data > limit] = channel_median
    data[data < -limit] = channel_median

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
            y = bandpass_filter(y, freqs=[250, 10e3], fs=fs, order=4)

        if len(lims) > 0:
            y = y[lims[0]:lims[1]]
            x = data["Time"][lims[0]:lims[1]]
        
        print(f"Plotting {column}...")
        fig.add_trace(go.Scatter(x=x, y=y+offset, mode='lines', name=column))
        
        offset += 10  # Max amplitude, peak-to-peak

    fig.show()


def vsr(data, fs, du, v_range, repeat, filt=True, chs_flipped=False, inv_polarity=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ 
    Velocity-selective recording (VSR) in the frequency domain with delay-and-add (B. W. Metcalfe, 2018)

    --> Not being used at the moment! <--

    """
    print(f"Computing delay-and-add for repeat {repeat}...")

    data = data[:, ::-1]
    
    # Number of velocities
    nv = len(v_range)

    # Get the length of the recordings and the number of channels
    nt, nu = data.shape
        
    # Frequency axis
    f = (np.arange(-nt/2, nt/2) / nt) * fs
    
    # Generate element positions
    u = np.arange(0, nu) * du
    urep = np.tile(u, [nt, 1])
        
    dft = signal.fft(data.T).T
    s = 1 / v_range

    im = np.zeros((len(data), len(v_range)), dtype=complex)

    for n in range(0, nv):
        delays = urep * s[n]

        # Delay
        delayed_data = np.exp(-1j * 2 * np.pi * np.tile(f.T, [nu, 1]).T * delays)
        shifted_fft = signal.fftshift(dft, 0)
        imn = signal.ifft(signal.ifftshift(shifted_fft * delayed_data, 0), axis=0)

        im[:, n] = np.sum(imn.T, axis=0)

    im = abs(im)

    # Now find the largest values for each delay
    samples = len(data)
    steps = len(v_range)

    largest = np.zeros(steps)
    smallest = np.zeros(steps)

    for i in range(steps):
        largest[i] = max(im[0:samples, i])
        smallest[i] = abs(min(im[0:samples, i]))

    print(f"Delay-and-add complete. (repeat {repeat})")

    return v_range, im, largest, smallest
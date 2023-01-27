import numpy as np
import pandas as pd
import torch

from scipy.interpolate import make_interp_spline
from scipy.signal import butter, filtfilt, find_peaks, resample
from sklearn.preprocessing import StandardScaler


def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def extract_resp_envelope(bp_data, device, num_chs=9):
    """Extract respiratory envelope from blood pressure window.

    Args: 
        bp_data: Blood pressure recording with shape (N,) (since same for all channels).
    Returns:
        resp_envelope: Resulting respiratory envelope.
    """
    # TODO: Needs adjusting and plots checking, or more smoothing?
    peaks, _ = find_peaks(bp_data, prominence=0.5)

    # TODO: Check if line below is needed here
    # peaks_x = [x/fs for x in peaks]

    # Create spline
    spl = make_interp_spline(peaks, bp_data[peaks])
    xnew = np.linspace(min(peaks), max(peaks), 1000)
    spl_y = spl(xnew)

    # Lowpass filter envelope so it's smoother
    resp_envelope = butter_lowpass_filter(spl_y, 15, 1000)

    resp_envelope = torch.from_numpy(resp_envelope.copy()).to(device).float()
    resp_envelope.repeat(num_chs)

    return resp_envelope


def compute_moving_rms(eng_data, bp_envelope, device, fs=100e3, rms_win_len=100e3, resample_num=1000, filter_cutoff=15):
    """
    tbd.
    """
    # First process BP data
    bp_envelope = prep_bp_data(bp_envelope)

    moving_rms_data = np.zeros((len(eng_data), resample_num))

    # Then ENG data
    for ch in range(len(eng_data)):
        # Take rolling RMS window of ENG data
        eng_rms = rolling_rms(eng_data[ch, :], rms_win_len)

        # Standardise RMS data (& drop NaNs from ENG data)
        scaler = StandardScaler()
        eng_rms_norm = scaler.fit_transform(eng_rms.dropna().to_numpy().reshape(-1, 1))

        # Flatten data
        eng_rms_norm = eng_rms_norm.flatten()

        # Resample and LPF ENG data
        moving_rms_data[ch, :] = butter_lowpass_filter(resample(eng_rms_norm, resample_num), filter_cutoff, resample_num)

    bp_envelope = torch.from_numpy(bp_envelope).to(device).float()
    moving_rms_data = torch.from_numpy(moving_rms_data).to(device).float()
    
    return bp_envelope, moving_rms_data


def prep_bp_data(bp_envelope):
    """
    tbd.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(bp_envelope.reshape(-1, 1))


def rolling_rms(x, N):
    # Source: https://dsp.stackexchange.com/a/74822
    return (pd.DataFrame(abs(x)**2).rolling(N).mean()) ** 0.5
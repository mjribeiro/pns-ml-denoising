import numpy as np
import pandas as pd
import torch
import torch.fft as fft

from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, filtfilt, find_peaks, resample
from sklearn.preprocessing import StandardScaler



def scaler_torch(x):
    mu = torch.mean(x, 0, keepdim=True)
    sd = torch.std(x, 0, unbiased=False, keepdim=True).clone()
    x = x - mu
    x = x / sd
    return x


def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


# Source: https://pytorch.org/blog/the-torch.fft-module-accelerated-fast-fourier-transforms-with-autograd-in-pyTorch/
def lowpass_torch(input, limit, device):
    pass1 = torch.abs(fft.rfftfreq(input.shape[-1])) < limit
    pass1 = pass1.to(device)
    # pass2 = torch.abs(fft.fftfreq(input.shape[-2])) < limit
    # kernel = torch.outer(pass2, pass1)
    
    fft_input = fft.rfft(input)
    return fft.irfft(fft_input * pass1)


# Source: https://stackoverflow.com/a/14314054
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def extract_resp_envelope(bp_data, len_data, device, num_chs=9):
    """Extract respiratory envelope from blood pressure window.

    Args: 
        bp_data: Blood pressure recording with shape (N,) (since same for all channels).
    Returns:
        resp_envelope: Resulting respiratory envelope.
    """
    # TODO: Needs adjusting and plots checking, or more smoothing?
    peaks, _ = find_peaks(bp_data, prominence=0.5, width=5000)

    # TODO: Check if line below is needed here
    # peaks_x = [x/fs for x in peaks]

    # Create spline
    # spl = make_interp_spline(peaks, bp_data[peaks], k=2)
    spl = PchipInterpolator(peaks, bp_data[peaks])

    xnew = np.linspace(min(peaks), max(peaks), len_data)
    spl_y = spl(xnew)
    # spl_y = moving_average(spl_y, n=100)

    # Lowpass filter envelope so it's smoother
    # resp_envelope = butter_lowpass_filter(spl_y, 15, 100e3)
    resp_envelope = spl_y

    # Prep BP data here
    resp_envelope = prep_bp_data(resp_envelope)

    # resp_envelope = torch.from_numpy(resp_envelope.copy()).to(device).float()
    resp_envelope = np.repeat(resp_envelope, num_chs, axis=1)
    resp_envelope = np.swapaxes(resp_envelope, 0, 1)

    return torch.tensor(resp_envelope).to(device).float()


def compute_moving_rms(eng_data, bp_envelope, device, fs=100e3, rms_win_len=int(100e3), resample_num=1000, filter_cutoff=10):
    """
    tbd.
    """
    # Convert ENG data from torch tensor to numpy
    # eng_data = eng_data.cpu().detach().numpy()

    moving_rms_data = torch.zeros((len(eng_data), resample_num - rms_win_len)).to(device)
    # moving_rms_data = np.zeros((len(eng_data), resample_num))

    # Then ENG data
    for ch in range(len(eng_data)):
        # Take rolling RMS window of ENG data
        # eng_rms_detached = rolling_rms(eng_data[ch, :].cpu().detach().numpy(), rms_win_len)
        eng_rms = rolling_rms_torch(eng_data[ch, :], rms_win_len)

        # Standardise RMS data (& drop NaNs from ENG data)
        # scaler = StandardScaler()
        # eng_rms_norm = scaler.fit_transform(eng_rms.reshape(-1, 1))
        eng_rms_norm = scaler_torch(eng_rms)

        # Flatten data
        eng_rms_norm = eng_rms_norm.flatten()

        # Update: no need to resample ENG data, spline for BP envelope was set to ENG # samples instead
        # resample_torch = Resample.apply
        # resampled_eng_rms = resample_torch(eng_rms_norm, resample_num)
        # moving_rms_data[ch, :] = butter_lowpass_filter(resample(eng_rms_norm, resample_num), filter_cutoff, 100)
        moving_rms_data[ch, :] = lowpass_torch(eng_rms_norm, filter_cutoff, device=device)

    # bp_envelope = torch.tensor(bp_envelope, requires_grad=True).to(device).float()
    # moving_rms_data = torch.tensor(moving_rms_data, requires_grad=True).to(device).float()
    
    return bp_envelope, moving_rms_data


def get_rms_envelope(x_hat, bp, window_len, device):
    """tbd
        Args:
        Outputs:
        Raises:
    """
    # Flatten input data, predictions, and blood pressure window per channel (since dataset is unshuffled)
    x_hat_flat = torch.flatten(torch.swapaxes(x_hat, 0, 1), start_dim=1)
    bp_flat = torch.flatten(torch.swapaxes(bp, 0, 1), start_dim=1)

    # Get dimensions and window sizes for moving RMS data
    len_data = x_hat_flat.shape[-1]
    window_len = int(100e3)
    len_envelope = len_data - window_len

    # Extract respiratory envelope from blood pressure data
    # Use channel 1 only since same for all channels
    bp_envelope = extract_resp_envelope(bp_flat[0, :], len_data=len_envelope, device=device, num_chs=9)

    # Get moving RMS plots
    bp_envelope, x_hat_moving_rms = compute_moving_rms(x_hat_flat, bp_envelope, device=device, fs=100e3, rms_win_len=window_len, resample_num=len_data)

    return bp_envelope, x_hat_moving_rms


def prep_bp_data(bp_envelope):
    """
    tbd.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(bp_envelope.reshape(-1, 1))


def rolling_rms(x, N):
    # Source: https://dsp.stackexchange.com/a/74822
    return (pd.DataFrame(abs(x)**2).rolling(N).mean()) ** 0.5


def rolling_rms_torch(x, N):
    # Source: https://dsp.stackexchange.com/a/74822
    xc = torch.cumsum(abs(x)**2, dim=0)
    return torch.sqrt((xc[N:] - xc[:-N])/ N)
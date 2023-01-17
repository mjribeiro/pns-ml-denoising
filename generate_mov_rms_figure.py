import matplotlib.figure 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import ticker as mtick
from scipy.interpolate import make_interp_spline
from scipy.signal import butter, filtfilt, find_peaks, resample
from sklearn.preprocessing import StandardScaler

# User-defined parameters
ch_num = 1
fs = 100e3

# Local imports
from utils.preprocessing import *
from datasets.vagus_dataset import VagusDatasetN2N

# Function definitions
def rolling_rms(x, N):
    # Source: https://dsp.stackexchange.com/a/74822
    return (pd.DataFrame(abs(x)**2).rolling(N).mean()) ** 0.5 

def calculate_envelope_points(x):
    peak_idx, _ = find_peaks(x, prominence=0.5)

    # Then return these and plot as line
    return peak_idx

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Load original test data
test_dataset = VagusDatasetN2N(stage="test")

# Load coordinate vae results
cvae_noisy_inputs_ch = np.load(f"./results/cvae_noisy_input_ch{ch_num}.npy")
cvae_reconstr_ch = np.load(f"./results/cvae_reconstr_ch{ch_num}.npy")

# Load n2n results
n2n_noisy_inputs_ch = np.load(f"./results/n2n_noisy_input_ch{ch_num}.npy")
n2n_noisy_targets_ch = np.load(f"./results/n2n_noisy_labels_ch{ch_num}.npy")
n2n_reconstr_ch = np.load(f"./results/n2n_reconstr_ch{ch_num}.npy")


# Make the following plots:
# 1) Original data and results with bandpass filtering
# 2) CVAE results overlaid on 50% opacity original
# 3) N2N results overlaid on 50% opacity original

# On a new plot:
# 1) Blood pressure with envelope
# 2) 20s recordings with moving RMS average similar to Metcalfe 2014 paper

w, h = matplotlib.figure.figaspect(1/4)
matplotlib.rcParams.update({'font.size': 16})
fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(w+2, h))
fig1.tight_layout()

w2, h2 = matplotlib.figure.figaspect(1/5)
matplotlib.rcParams.update({'font.size': 16})
fig2, (ax4, ax5, ax6, ax7) = plt.subplots(1,4, figsize=(w2 + 2, h2))
fig2.tight_layout()

# Indices for zommed in plots
start_factor = 0
end_factor = 5

zoom_start = start_factor
zoom_end = 1024 * end_factor

zoom_cvae_noisy = cvae_noisy_inputs_ch[zoom_start:zoom_end]
zoom_cvae_reconstr = cvae_reconstr_ch[zoom_start:zoom_end]

zoom_n2n_noisy_inputs = n2n_noisy_inputs_ch[zoom_start:zoom_end]
zoom_n2n_noisy = n2n_noisy_targets_ch[zoom_start:zoom_end]
zoom_n2n_reconstr = n2n_reconstr_ch[zoom_start:zoom_end]

zoom_time = np.arange(0, len(zoom_cvae_noisy)/fs, 1/fs)

# 1) Original plus bandpass filtered
orig_filtered_win = bandpass_filter(zoom_n2n_noisy_inputs, [250, 10e3], fs)
orig_filtered = bandpass_filter(n2n_noisy_inputs_ch, [250, 10e3], fs)

ax1.plot(zoom_time, zoom_n2n_noisy_inputs, alpha=0.3)
ax1.plot(zoom_time, orig_filtered_win)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude (norm.)")
ax1.set_ylim([-1, 1])

# 2) CVAE data
ax2.plot(zoom_time, zoom_cvae_noisy, alpha=0.3)
ax2.plot(zoom_time, zoom_cvae_reconstr)
ax2.set_xlabel("Time (s)")
# ax2.set_ylabel("Amplitude (norm.)")
ax2.set_ylim([-1, 1])

# 3) N2N data
ax3.plot(zoom_time, zoom_n2n_noisy, alpha=0.3)
ax3.plot(zoom_time, zoom_n2n_reconstr)
ax3.set_xlabel("Time (s)")
# ax3.set_ylabel("Amplitude (norm.)")
ax3.set_ylim([-1, 1])

fig1.savefig("./plots/denoised.png", bbox_inches='tight')

# 4) BP with envelope
entire_bp_recording = test_dataset.load_bp_data(0, len(test_dataset))

# Use only data that was used for testing (drop_last=True when testing, so not entire dataset was used)
time_full = np.arange(0, len(cvae_reconstr_ch)/fs, 1/fs)
max_idx = len(time_full)

ch1_bp_recording = entire_bp_recording[:, 0, :].flatten()

peaks = calculate_envelope_points(ch1_bp_recording)
peaks = list(filter(lambda x : x < max_idx, peaks))

peaks_x = [x/fs for x in peaks]

# Make spline from envelope points
spl = make_interp_spline(peaks_x, ch1_bp_recording[peaks])
xnew = np.linspace(min(peaks_x), max(peaks_x), 1000)
spl_y = spl(xnew)

# Lowpass filter envelope so it's smoother
bp_envelope = butter_lowpass_filter(spl_y, 15, 1000)

ax4.plot(time_full, ch1_bp_recording[:max_idx])
ax4.plot(xnew, bp_envelope)
ax4.set_xlim([-1.5, 21.5])
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Amplitude")

# 5) Moving RMS window of bandpass filtered data
N_BANDPASS = int(1000e-3 * fs) # 1000 ms in samples
bandpass_rms = rolling_rms(orig_filtered, N_BANDPASS)
ax5.plot(time_full, bandpass_rms)
ax5.set_xlim([-1.5, 21.5])
ax5.set_xlabel("Time (s)")
# ax5.set_ylabel("Amplitude (norm.)")

# 6) Moving RMS window of whole data
N_CVAE = int(1000e-3 * fs) # 1000 ms in samples
N_N2N = int(1000e-3 * fs) # 1000 ms in samples

# 6) CVAE
cvae_reconstr_ch_rms = rolling_rms(cvae_reconstr_ch, N_CVAE) 
ax6.plot(time_full, cvae_reconstr_ch_rms)
ax6.set_xlim([-1.5, 21.5])
ax6.set_xlabel("Time (s)")
# ax6.set_ylabel("Amplitude (norm.)")

# 7) N2N
n2n_reconstr_ch_rms = rolling_rms(n2n_reconstr_ch, N_N2N) 
ax7.plot(time_full, n2n_reconstr_ch_rms)
ax7.set_xlim([-1.5, 21.5])
ax7.set_xlabel("Time (s)")
# ax7.set_ylabel("Amplitude (norm.)")

plt.savefig("./plots/mov_rms.png", bbox_inches='tight')

# TO TRY
# - envelopes for raw data and bandpass filtered
# - VAE with bandpass filtered data (as per original paper)
# - VAE output as Noise2Noise targets

# - Multiple ch_nums as multiple Noise2Noise pairs (different noise on each ch_num, maybe use adjacent pairs but look at amplitudes in spontaneous data first, adjust for time difference)

scaler = StandardScaler()
bp_envelope_norm = scaler.fit_transform(bp_envelope.reshape(-1, 1))

bandpass_rms_norm = scaler.fit_transform(bandpass_rms.dropna().to_numpy().reshape(-1, 1))
cvae_reconstr_ch1_rms_norm = scaler.fit_transform(cvae_reconstr_ch_rms.dropna().to_numpy().reshape(-1, 1))
n2n_reconstr_ch1_rms_norm = scaler.fit_transform(n2n_reconstr_ch_rms.dropna().to_numpy().reshape(-1, 1))

bandpass_x_corr = resample(bandpass_rms_norm, len(bp_envelope_norm)).ravel()
cvae_x_corr = resample(cvae_reconstr_ch1_rms_norm, len(bp_envelope_norm)).ravel()
n2n_x_corr = resample(n2n_reconstr_ch1_rms_norm, len(bp_envelope_norm)).ravel()

print("Bandpass x-correlation: ", calculate_cross_correlation(bp_envelope_norm.ravel(), bandpass_x_corr)[0,1])
print("VAE x-correlation: ", calculate_cross_correlation(bp_envelope_norm.ravel(), cvae_x_corr)[0,1])
print("N2N x-correlation: ", calculate_cross_correlation(bp_envelope_norm.ravel(), n2n_x_corr)[0,1])

# Find peaks in envelopes, see which matches the right frequency - 0.25 Hz
respiratory_freq = 0.25
bandpass_rms_peaks, _ = find_peaks(bandpass_rms.to_numpy().ravel(), distance=3*fs)
cvae_rms_peaks, _ = find_peaks(cvae_reconstr_ch_rms.to_numpy().ravel(), distance=3*fs)
n2n_rms_peaks, _ = find_peaks(n2n_reconstr_ch_rms.to_numpy().ravel(), distance=3*fs)

# Find mean spacing between peaks, return as frequency (1/ time differences) to compare with 0.25 Hz
bandpass_peak_spacing = np.mean( 1 / (np.diff(bandpass_rms_peaks) / fs))
cvae_peak_spacing = np.mean( 1 / (np.diff(cvae_rms_peaks) / fs))
n2n_peak_spacing = np.mean( 1 / (np.diff(n2n_rms_peaks) / fs))

bandpass_freq_diff = ((respiratory_freq - bandpass_peak_spacing) / respiratory_freq) * 100
cvae_freq_diff = ((respiratory_freq - cvae_peak_spacing) / respiratory_freq) * 100
n2n_freq_diff = ((respiratory_freq - n2n_peak_spacing) / respiratory_freq) * 100

print("Bandpass percent diff in resp. rate: ", bandpass_freq_diff)
print("VAE percent diff in resp. rate: ", cvae_freq_diff)
print("N2N percent diff in resp. rate: ", n2n_freq_diff)

# Try and work out an approximate SNR?
# Take section of baseline as approximate background noise, inspect plots manually first to see where these sections are
num_samples = 120
bandpass_start = int(5.28 * fs)
cvae_start = int(0.0026 * fs)
n2n_start = int(0.88 * fs)

bandpass_end = bandpass_start + num_samples
cvae_end = cvae_start + num_samples
n2n_end = n2n_start + num_samples

bandpass_baseline_sample = orig_filtered[bandpass_start:bandpass_end]
cvae_baseline_sample = cvae_reconstr_ch[cvae_start:cvae_end]
n2n_baseline_sample = n2n_reconstr_ch[n2n_start:n2n_end]

# Take variance to get average power? (N)
bandpass_baseline_noise_pwr = np.var(bandpass_baseline_sample)
cvae_baseline_noise_pwr = np.var(cvae_baseline_sample)
n2n_baseline_noise_pwr = np.var(n2n_baseline_sample)

# Take variance of signal as average power (S = NoisyS - N?)
bandpass_signal_pwr = np.var(orig_filtered) - bandpass_baseline_noise_pwr
cvae_signal_pwr = np.var(cvae_reconstr_ch) - cvae_baseline_noise_pwr
n2n_signal_pwr = np.var(n2n_reconstr_ch) - n2n_baseline_noise_pwr

# Find SNR in dB (10log_10(S/N))
bandpass_snr = 10 * np.log10(bandpass_signal_pwr / bandpass_baseline_noise_pwr)
cvae_snr = 10 * np.log10(cvae_signal_pwr / cvae_baseline_noise_pwr)
n2n_snr = 10 * np.log10(n2n_signal_pwr / n2n_baseline_noise_pwr)

print("Bandpass SNR: ", bandpass_snr)
print("VAE SNR: ", cvae_snr)
print("N2N SNR: ", n2n_snr)
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import ticker as mtick
from scipy.interpolate import make_interp_spline, PchipInterpolator
from scipy.signal import butter, filtfilt, find_peaks, resample
from sklearn.preprocessing import StandardScaler

# User-defined parameters
num_models = 3
select_ch = 0
num_chs = 9
fs = 100e3

# Local imports
from utils.preprocessing import *
from datasets.vagus_dataset import VagusDatasetN2N

# Function definitions
def rolling_rms(x, N):
    # Source: https://dsp.stackexchange.com/a/74822
    return (pd.DataFrame(abs(x)**2).rolling(N).mean()) ** 0.5

def calculate_envelope_points(x):
    peak_idx, _ = find_peaks(x, prominence=0.5, width=5000)

    # Then return these and plot as line
    return peak_idx

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low')

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Load original test data
test_dataset = VagusDatasetN2N(stage="test")

# Load coordinate vae results
print("Starting to load all channel data...")
for ch in range(num_chs):
    # Initialise channel data arrays on first input since data length not known a priori
    if ch == 0:
        # Initialise VAE arrays
        vae_noisy_inputs_ch = np.load(f"./results/cvae_noisy_input_ch{ch+1}.npy")
        vae_reconstr_ch     = np.load(f"./results/cvae_reconstr_rms_ch{ch+1}.npy")

        vae_noisy_inputs = np.zeros((len(vae_noisy_inputs_ch), num_chs))
        vae_reconstr     = np.zeros((len(vae_reconstr_ch), num_chs))

        # Assign first set of data
        vae_noisy_inputs[:, ch] = vae_noisy_inputs_ch
        vae_reconstr[:, ch]     = vae_reconstr_ch

        # Initialise N2N arrays
        n2n_noisy_inputs_ch  = np.load(f"./results/n2n_noisy_input_ch{ch+1}.npy")
        n2n_noisy_targets_ch = np.load(f"./results/n2n_noisy_labels_ch{ch+1}.npy")
        n2n_reconstr_ch      = np.load(f"./results/n2n_reconstr_ch{ch+1}.npy")

        n2n_noisy_inputs  = np.zeros((len(n2n_noisy_inputs_ch), num_chs))
        n2n_noisy_targets = np.zeros((len(n2n_noisy_targets_ch), num_chs))
        n2n_reconstr      = np.zeros((len(n2n_reconstr_ch), num_chs))

        # Assign first set of data
        n2n_noisy_inputs[:, ch]  = n2n_noisy_inputs_ch
        n2n_noisy_targets[:, ch] = n2n_noisy_targets_ch
        n2n_reconstr[:, ch]      = n2n_reconstr_ch

    else:
        # VAE data
        vae_noisy_inputs[:, ch] = np.load(f"./results/cvae_noisy_input_ch{ch+1}.npy")
        vae_reconstr[:, ch]     = np.load(f"./results/cvae_reconstr_rms_ch{ch+1}.npy")

        # N2N data
        n2n_noisy_inputs[:, ch]  = np.load(f"./results/n2n_noisy_input_ch{ch+1}.npy")
        n2n_noisy_targets[:, ch] = np.load(f"./results/n2n_noisy_labels_ch{ch+1}.npy")
        n2n_reconstr[:, ch]      = np.load(f"./results/n2n_reconstr_ch{ch+1}.npy")

print("Finished loading all channel data.\n")

# --- GENERATING PLOTS
# Make the following plots:
# 1) Original data and results with bandpass filtering
# 2) VAE results overlaid on 50% opacity original
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
start_factor = 320
end_factor = 480

zoom_start = 1024 * start_factor
zoom_end = 1024 * end_factor

zoom_vae_noisy = vae_noisy_inputs[zoom_start:zoom_end, select_ch]
zoom_vae_reconstr = vae_reconstr[zoom_start:zoom_end, select_ch]

zoom_n2n_noisy_inputs = n2n_noisy_inputs[zoom_start:zoom_end, select_ch]
zoom_n2n_noisy = n2n_noisy_targets[zoom_start:zoom_end, select_ch]
zoom_n2n_reconstr = n2n_reconstr[zoom_start:zoom_end, select_ch]

zoom_time = np.arange(0, len(zoom_vae_noisy)/fs, 1/fs)

# 1) Original plus bandpass filtered
orig_filtered_win = bandpass_filter(zoom_n2n_noisy_inputs, [250, 10e3], fs)
orig_filtered = bandpass_filter(n2n_noisy_inputs[:, select_ch], [250, 10e3], fs)

ax1.plot(zoom_time, zoom_n2n_noisy_inputs, alpha=0.3)
ax1.plot(zoom_time, orig_filtered_win)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude (norm.)")
ax1.set_ylim([-1.2, 1.2])

# 2) VAE data
ax2.plot(zoom_time, zoom_vae_noisy, alpha=0.3)
ax2.plot(zoom_time, zoom_vae_reconstr)
ax2.set_xlabel("Time (s)")
# ax2.set_ylabel("Amplitude (norm.)")
ax2.set_ylim([-1.2, 1.2])

# 3) N2N data
ax3.plot(zoom_time, zoom_n2n_noisy, alpha=0.3)
ax3.plot(zoom_time, zoom_n2n_reconstr)
ax3.set_xlabel("Time (s)")
# ax3.set_ylabel("Amplitude (norm.)")
ax3.set_ylim([-1.2, 1.2])

fig1.savefig("./plots/denoised.png", bbox_inches='tight')
# fig1.show()

# 4) BP with envelope
entire_bp_recording = test_dataset.load_bp_data(0, len(test_dataset))

# Use only data that was used for testing (drop_last=True when testing, so not entire dataset was used)
time_full = np.arange(0, len(vae_reconstr[:, select_ch])/fs, 1/fs)
max_idx = len(time_full)

ch1_bp_recording = entire_bp_recording[:, 0, :].flatten()
# ch1_bp_recording = butter_lowpass_filter(ch1_bp_recording, 100, 100e3)

peaks = calculate_envelope_points(ch1_bp_recording)
peaks = list(filter(lambda x : x < max_idx, peaks))

peaks_x = [x/fs for x in peaks]

# Make spline from envelope points
spl = make_interp_spline(peaks_x, ch1_bp_recording[peaks])
# spl = PchipInterpolator(peaks_x, ch1_bp_recording[peaks])
# xnew = np.linspace(min(peaks_x), max(peaks_x), len(vae_reconstr[:, select_ch]))
xnew = np.linspace(min(peaks_x), max(peaks_x), 1000)
spl_y = spl(xnew)

# Lowpass filter envelope so it's smoother
bp_envelope = butter_lowpass_filter(spl_y, 15, 1000)
# bp_envelope = spl_y

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
N_VAE = int(1000e-3 * fs) # 1000 ms in samples
N_N2N = int(1000e-3 * fs) # 1000 ms in samples

# 6) VAE
vae_reconstr_ch_rms = rolling_rms(vae_reconstr[:, select_ch], N_VAE)
ax6.plot(time_full, vae_reconstr_ch_rms)
ax6.set_xlim([-1.5, 21.5])
ax6.set_xlabel("Time (s)")
# ax6.set_ylabel("Amplitude (norm.)")

# 7) N2N
n2n_reconstr_ch_rms = rolling_rms(n2n_reconstr[:, select_ch], N_N2N)
ax7.plot(time_full, n2n_reconstr_ch_rms)
ax7.set_xlim([-1.5, 21.5])
ax7.set_xlabel("Time (s)")
# ax7.set_ylabel("Amplitude (norm.)")

plt.savefig("./plots/mov_rms.png", bbox_inches='tight')
# plt.show()


# --- EXTRACTING METRICS
# Normalise both BP envelope and RMS data before performing cross correlation

scaler = StandardScaler()
bp_envelope_norm = scaler.fit_transform(bp_envelope.reshape(-1, 1))

# Resampling to 1000 long signals to match blood pressure envelope length for cross-correlation
resample_num = len(bp_envelope_norm)
# resample_num = len(bp_envelope_norm) - int(fs) + 1

dx = (1 / fs) * len(bandpass_rms) / resample_num
new_fs = 1 / dx

filter_cutoff = 1

# Ground-truth respiratory rate
respiratory_freq = 0.25

cross_corr_waveforms = np.zeros((num_models, num_chs, resample_num))
cross_corr_all_chs   = np.zeros((num_models, num_chs))
respir_rate_all_chs  = np.zeros((num_models, num_chs))
mse_all_chs  = np.zeros((num_models, num_chs))


for ch in range(num_chs):
    # bandpass_data = bandpass_filter(n2n_noisy_inputs[:, ch], [250, 10e3], fs)
    bandpass_data = n2n_noisy_targets[:, ch]

    # Moving RMS window for specific channel
    bandpass_rms     = rolling_rms(bandpass_data, N_BANDPASS)
    vae_reconstr_rms = rolling_rms(vae_reconstr[:, ch], N_VAE)
    n2n_reconstr_rms = rolling_rms(n2n_reconstr[:, ch], N_N2N)

    # Standardise and drop NaNs from channel data
    bandpass_rms_norm     = scaler.fit_transform(bandpass_rms.dropna().to_numpy().reshape(-1, 1))
    vae_reconstr_rms_norm = scaler.fit_transform(vae_reconstr_rms.dropna().to_numpy().reshape(-1, 1))
    n2n_reconstr_rms_norm = scaler.fit_transform(n2n_reconstr_rms.dropna().to_numpy().reshape(-1, 1))

    # Normalised RMS windows
    bandpass_rms_norm     = bandpass_rms_norm.flatten()
    vae_reconstr_rms_norm = vae_reconstr_rms_norm.flatten()
    n2n_reconstr_rms_norm = n2n_reconstr_rms_norm.flatten()

    # Resample to same length and lowpass filter
    bandpass_x_corr = butter_lowpass_filter(resample(bandpass_rms_norm, resample_num), filter_cutoff, new_fs)
    vae_x_corr      = butter_lowpass_filter(resample(vae_reconstr_rms_norm, resample_num), filter_cutoff, new_fs)
    n2n_x_corr      = butter_lowpass_filter(resample(n2n_reconstr_rms_norm, resample_num), filter_cutoff, new_fs)
    # bandpass_x_corr = butter_lowpass_filter(bandpass_rms_norm, filter_cutoff, fs)
    # vae_x_corr      = butter_lowpass_filter(vae_reconstr_rms_norm, filter_cutoff, fs)
    # n2n_x_corr      = butter_lowpass_filter(n2n_reconstr_rms_norm, filter_cutoff, fs)

    # Store waveforms for plotting later
    cross_corr_waveforms[0, ch, :] = bandpass_x_corr
    cross_corr_waveforms[1, ch, :] = vae_x_corr
    cross_corr_waveforms[2, ch, :] = n2n_x_corr

    # CROSS-CORRELATION
    cross_corr_all_chs[0, ch] = np.corrcoef(bp_envelope_norm.ravel(), bandpass_x_corr)[0,1]
    cross_corr_all_chs[1, ch] = np.corrcoef(bp_envelope_norm.ravel(), vae_x_corr)[0,1]
    cross_corr_all_chs[2, ch] = np.corrcoef(bp_envelope_norm.ravel(), n2n_x_corr)[0,1]
    # cross_corr_all_chs[0, ch] = np.corrcoef(resample(bp_envelope_norm, resample_num).ravel(), bandpass_x_corr)[0,1]
    # cross_corr_all_chs[1, ch] = np.corrcoef(resample(bp_envelope_norm, resample_num).ravel(), vae_x_corr)[0,1]
    # cross_corr_all_chs[2, ch] = np.corrcoef(resample(bp_envelope_norm, resample_num).ravel(), n2n_x_corr)[0,1]


    # CHECKING RESPIRATORY RATE
    # Using same inputs as x-corr, so need to use modified fs since resampling took place
    bandpass_rms_peaks, _ = find_peaks(bandpass_x_corr, prominence=0.8)
    vae_rms_peaks, _      = find_peaks(vae_x_corr, prominence=0.8)
    n2n_rms_peaks, _      = find_peaks(n2n_x_corr, prominence=0.8)

    # Find mean spacing between peaks, return as frequency (1/ time differences) to compare with 0.25 Hz
    bandpass_peak_spacing = np.mean( 1 / (np.diff(bandpass_rms_peaks) / new_fs))
    vae_peak_spacing      = np.mean( 1 / (np.diff(vae_rms_peaks) / new_fs))
    n2n_peak_spacing      = np.mean( 1 / (np.diff(n2n_rms_peaks) / new_fs))

    respir_rate_all_chs[0, ch] = abs((respiratory_freq - bandpass_peak_spacing) / respiratory_freq) * 100
    respir_rate_all_chs[1, ch] = abs((respiratory_freq - vae_peak_spacing) / respiratory_freq) * 100
    respir_rate_all_chs[2, ch] = abs((respiratory_freq - n2n_peak_spacing) / respiratory_freq) * 100

    # CHECKING MSE
    mse_all_chs[0, ch] = ((bp_envelope_norm.ravel() - bandpass_x_corr) ** 2).mean(axis=0)
    mse_all_chs[1, ch] = ((bp_envelope_norm.ravel() - vae_x_corr) ** 2).mean(axis=0)
    mse_all_chs[2, ch] = ((bp_envelope_norm.ravel() - n2n_x_corr) ** 2).mean(axis=0)


print(f"Bandpass x-correlation \t MEAN: {np.mean(cross_corr_all_chs[0, :])} \t STD: {np.std(cross_corr_all_chs[0, :])}")
print(f"VAE x-correlation \t MEAN: {np.mean(cross_corr_all_chs[1, :])} \t STD: {np.std(cross_corr_all_chs[1, :])}")
print(f"N2N x-correlation \t MEAN: {np.mean(cross_corr_all_chs[2, :])} \t STD: {np.std(cross_corr_all_chs[2, :])}")

print(f"Bandpass percent diff in resp. rate \t MEAN: {np.mean(respir_rate_all_chs[0, :])} \t STD: {np.std(respir_rate_all_chs[0, :])}")
print(f"VAE percent diff in resp. rate \t\t MEAN: {np.mean(respir_rate_all_chs[1, :])} \t STD: {np.std(respir_rate_all_chs[1, :])}")
print(f"N2N percent diff in resp. rate \t\t MEAN: {np.mean(respir_rate_all_chs[2, :])} \t STD: {np.std(respir_rate_all_chs[2, :])}")

print(f"Bandpass MSE \t MEAN: {np.mean(mse_all_chs[0, :])} \t STD: {np.std(mse_all_chs[0, :])}")
print(f"VAE MSE \t\t MEAN: {np.mean(mse_all_chs[1, :])} \t STD: {np.std(mse_all_chs[1, :])}")
print(f"N2N MSE \t\t MEAN: {np.mean(mse_all_chs[2, :])} \t STD: {np.std(mse_all_chs[2, :])}")




# ----- Extra plots for debugging -----
# Blood pressure with envelope
plt.figure()
plt.plot(time_full, ch1_bp_recording[:max_idx])
plt.plot(xnew, bp_envelope)
plt.show()

# Inputs to cross-correlation
matplotlib.rcParams.update({'font.size': 13})
time_xcorr = np.linspace(0, time_full[-1], len(bp_envelope_norm))
plt.figure()
plt.plot(time_xcorr, bp_envelope_norm, label="BP envelope", alpha=0.75, linestyle='-', linewidth=2)
plt.plot(time_xcorr, np.mean(cross_corr_waveforms[0, :, :], axis=0), label="Bandpass", alpha=1, linestyle='--', linewidth=2)
plt.plot(time_xcorr, np.mean(cross_corr_waveforms[1, :, :], axis=0), label="VAE", alpha=1, linestyle='-.', linewidth=2)
plt.plot(time_xcorr, np.mean(cross_corr_waveforms[2, :, :], axis=0), label="Noise2Noise", alpha=1, linestyle=':', linewidth=2)
plt.legend(loc="upper left", ncol=4, columnspacing=0.7, prop={'size': 10.5})
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (normalised)")
plt.savefig('./plots/mov_rms_overlap.png')


# Test
plt.figure()
plt.plot(resample(bandpass_rms_norm, resample_num))
plt.plot(cross_corr_waveforms[0, 8, :])
plt.savefig('./plots/bandpass_filter_check.png')

# # 16-17s section of test set ENG signal
# start_time_debug = int(15.8 * fs)
# end_time_debug = int(start_time_debug + (1.2 * fs))

# plt.figure()
# plt.plot(time_full[start_time_debug:end_time_debug], n2n_noisy_inputs[start_time_debug:end_time_debug, select_ch], label="Noisy data (N2N only)", alpha=0.5)
# plt.plot(time_full[start_time_debug:end_time_debug], n2n_noisy_targets[start_time_debug:end_time_debug, select_ch], label="BP-filtered data (N2N and VAE)", alpha=0.5)
# plt.plot(time_full[start_time_debug:end_time_debug], vae_reconstr[start_time_debug:end_time_debug, select_ch], label="VAE reconstr", alpha=0.5)
# plt.plot(time_full[start_time_debug:end_time_debug], n2n_reconstr[start_time_debug:end_time_debug, select_ch], label="N2N reconstr", alpha=0.5)
# plt.legend()
# plt.show()
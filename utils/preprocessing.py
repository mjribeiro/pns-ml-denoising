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
        # width=1200,
        # height=800,
        # font=dict(
        #     size=18,
        # ),
        # xaxis = dict(
        #     dtick = 1
        # )
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

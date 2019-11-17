import pandas as pd
import os
import numpy as np
from scripts.process import minus,plus
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator


def plot(data):
    try:
        assert(type(data)==dict)
    except Exception:
        data = {"unnamed":data}
    figs = {}
    for idx,datai in enumerate(data):
        fig = plt.figure("Plot EEG")
        # Load the EEG data
        n_samples, n_rows = data[datai].shape
        t = 10 * np.arange(n_samples) / n_samples
        # Plot the EEG
        ticklocs = []
        figs[idx] = fig.add_subplot(1,len(data.keys()), idx+1)
        figs[idx].set_xlim(0, 10)
        figs[idx].set_xticks(np.arange(10))
        figs[idx].set_title(datai)
        dmin = data[datai].min()
        dmax = data[datai].max()
        dr = (dmax - dmin) * 0.7  # Crowd them a bit.
        y0 = dmin
        y1 = (n_rows - 1) * dr + dmax
        figs[idx].set_ylim(y0, y1)
        segs = []
        for i in range(n_rows):
            segs.append(np.column_stack((t, data[datai][:, i])))    
            ticklocs.append(i * dr)
        offsets = np.zeros((n_rows, 2), dtype=float)
        offsets[:, 1] = ticklocs
        lines = LineCollection(segs, offsets=offsets, transOffset=None)
        figs[idx].add_collection(lines)
        figs[idx].set_yticks(ticklocs)
        figs[idx].set_xlabel('Time (s)')
        figs[idx].set_yticklabels(np.arange(0,22))

    plt.tight_layout()
    plt.show()
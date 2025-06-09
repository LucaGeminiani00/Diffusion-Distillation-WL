from scipy.interpolate import UnivariateSpline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_ts(data, column=None):
    """
    Plots time series for the specified column or all columns.
    
    Parameters:
        data (pd.DataFrame or dict-like): Input data to plot.
        column (str or None): Column name to plot. If None, plots all columns.
    """
    data = pd.DataFrame(data)
    

    if column and column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data columns: {data.columns.tolist()}")

    plt.figure(figsize=(12, 8))
    
    if column:
        plt.plot(data.index, data[column], label=column)
    else:
        for col in data.columns:
            plt.plot(data.index, data[col], label=col)

    plt.title(f"Time Series for {'Column: ' + column if column else 'All Columns'}")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()

from scipy.interpolate import UnivariateSpline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_smoothed_ts(data, column=None, smoothing_factor=2):
    """
    Plots smoothed time series for the specified column or all columns using spline interpolation.

    Parameters:
        data (pd.DataFrame or dict-like): Input data to plot.
        column (str or None): Column name to plot. If None, plots all columns.
        smoothing_factor (float): Smoothing factor for the spline. Higher values result in smoother curves.
    """
    data = pd.DataFrame(data)

    if column and column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data columns: {data.columns.tolist()}")

    plt.figure(figsize=(12, 8))

    if column:
        x = np.arange(len(data.index))
        y = data[column].values
        spline = UnivariateSpline(x, y, s=smoothing_factor)
        smoothed_y = spline(x)
        plt.plot(data.index, smoothed_y, label=f"Smoothed {column}")
    else:
        for col in data.columns:
            x = np.arange(len(data.index))
            y = data[col].values
            spline = UnivariateSpline(x, y, s=smoothing_factor)
            smoothed_y = spline(x)
            plt.plot(data.index, smoothed_y, label=f"Smoothed {col}")

    plt.title(f"Smoothed Time Series for {'Column: ' + column if column else 'All Columns'}")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()

def invert_windows(x):
    data = np.zeros((x.shape[0], x.shape[2]))  
    for i in range(x.shape[0]):
        data[i, :] = x[i, -1, :]  
    return data 


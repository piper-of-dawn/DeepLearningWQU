import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np

import pandas as pd
from statsmodels.tsa.stattools import adfuller

def adf_test(series, alpha=0.05):
    """
    Perform Augmented Dickey-Fuller (ADF) test for stationarity.

    Parameters:
    - series (pd.Series): Time series data.
    - alpha (float): Significance level for the test.

    Returns:
    - ADF test results and conclusion.
    """
    result = adfuller(series, autolag='AIC')
    p_value = result[1]

    if p_value <= alpha:
        return f"Result: Reject the null hypothesis at {alpha} significance level (p-value: {round(p_value,2)}). The time series is likely stationary."
    else:
        return f"Result: Fail to reject the null hypothesis at {alpha} significance level (p-value: {round(p_value,2)}). The time series is likely non-stationary."



def plot_acf_pacf_side_by_side(data, lags=None, padding=0.1, title='Autocorrelation and Partial Autocorrelation Functions'):
    # Set a neutral color palette and Arial font
    sns.set_palette("gray")
    plt.rcParams['font.family'] = 'Arial'

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # ACF plot
    plot_acf(data, lags=lags, ax=ax[0], color='gray')
    ax[0].set_title('Autocorrelation Function (ACF)')

    # PACF plot
    plot_pacf(data, lags=lags, ax=ax[1], color='gray')
    ax[1].set_title('Partial Autocorrelation Function (PACF)')

    # Remove spines from both x and y axes
    sns.despine(ax=ax[0], left=True, right=True, top=True, bottom=True)
    sns.despine(ax=ax[1], left=True, right=True, top=True, bottom=True)

    # Adjust layout with padding
    plt.subplots_adjust(wspace=padding)

    # Add footnote
    fig.text(0.5, -0.05, adf_test(data, 0.05), ha='center', fontsize=10, color='gray')
    plt.suptitle(title, y=1.02, fontsize=14, color='black',fontweight= 'bold')
    # Show the plots
    return fig

def plot_smooth_multiline_chart(data, title='Smooth Multiline Chart', x_label='X-axis', y_label='Y-axis'):
    # Define the color scheme with shades of red and black
    colors = ['#2D132C','#F95959','#801336','#233142','#455D7A','#C72C41','#EE4540','#E3E3E3']
    # Check if the number of data sets is within the supported range
    if len(data) > len(colors):
        raise ValueError("Too many data sets. Maximum supported is {}".format(len(colors)))

    # Create a plot with a different color for each data set
    plt.figure(figsize=(10, 6))
    for i, (x_axis, y_axis, label) in enumerate(data):
        plt.plot(x_axis, y_axis, label=label, color=colors[i])

    # Add labels and legend
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

   # Add mildly visible dotted grid lines in x direction
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Remove spines from x and y axis
    sns.despine(left=True, bottom=True)

    # Add legend
    plt.legend()

    return plt


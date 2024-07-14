import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import calplot
from statsmodels.graphics.tsaplots import plot_acf


def plot_variable_trend(dataset: pd.DataFrame, variable_to_plot: str) -> None:
    """Plot the time series trend of a variable (column) in a DataFrame.

    Args:
        dataset (pd.DataFrame): DataFrame containing the time series data.
        variable_to_plot (str): Name of the variable (column) to be plotted.

    Returns:
        None
    """
    plt.figure(figsize=(10, 3), tight_layout=True)
    plt.plot(dataset[variable_to_plot], label=variable_to_plot)
    plt.title(f'Time-Series of "{variable_to_plot}" Variable', weight='bold')
    plt.xlabel('Time [daily frequency]')
    plt.ylabel(f'{variable_to_plot}')
    plt.legend()
    plt.show()


def plot_variable_histogram(dataset: pd.DataFrame, variable_to_plot: str) -> None:
    """Plot the histogram of a variable (column) in a DataFrame to visualize its
    distribution.

    Args:
        dataset (pd.DataFrame): DataFrame containing the time series data.
        variable_to_plot (str): Name of the variable (column) to be plotted.

    Returns:
        None
    """
    plt.figure(figsize=(10, 3), tight_layout=True)
    plt.title(f'Histogram of "{variable_to_plot}" Variable', weight='bold')
    sns.histplot(data=dataset, x=variable_to_plot, kde=True, element='step')
    plt.show()


def plot_variable_boxplot_violinplot(dataset: pd.DataFrame, variable_to_plot: str) -> None:
    """Plot the Box-plot and Violin-plot of a variable in a DataFrame.

    Box Plot: Shows the distribution of the data with IQR (interquartile range), highlighting
        the outliers and the median value.
    Violin Plot: Shows the distribution of the data with a kernel density overlaid on a boxplot-like
        representation, providing a clearer view of the data density.

    Args:
        dataset (pd.DataFrame): DataFrame containing the time series data.
        variable_to_plot (str): Name of the variable (column) to be plotted.

    Returns:
        None
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=True, tight_layout=True)

    # plots
    b = sns.boxplot(y=dataset[variable_to_plot], ax=ax[0])
    v = sns.violinplot(y=dataset[variable_to_plot], ax=ax[1])

    # titles
    b = b.set_title(f'Box-plot "{variable_to_plot}" Data', weight='bold')
    v = v.set_title(f'Violin-plot "{variable_to_plot}" Data', weight='bold')
    plt.show()


def plot_variable_boxplot_violinplot_yearly(dataset: pd.DataFrame, variable_to_plot: str) -> None:
    """Plot the Box-plot and Violin-plot of a variable in a DataFrame across the years.

    Box Plot: Shows the distribution of the data with IQR (interquartile range), highlighting
        the outliers and the median value.
    Violin Plot: Shows the distribution of the data with a kernel density overlaid on a
        boxplot-like representation, providing a clearer view of the data density.

    Args:
        dataset (pd.DataFrame): DataFrame containing the time series data.
        variable_to_plot (str): Name of the variable (column) to be plotted.

    Returns:
        None
    """
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=True, sharey=True, tight_layout=True)
    fig.suptitle(f'Box-plot and Violin-plot Yearly Data Comparison of "{variable_to_plot}" Variable', weight='bold')

    # plots
    sns.boxplot(
        data=dataset, x=dataset.index.year, y=variable_to_plot,
        hue=dataset.index.year, ax=ax[0], legend=False, palette='deep'
    )
    sns.violinplot(
        data=dataset, x=dataset.index.year, y=variable_to_plot,
        hue=dataset.index.year, ax=ax[1], legend=False, palette='deep'
    )
    ax[0].set_xlabel('')
    plt.show()


def plot_variable_boxplot_monthly(dataset_2013: pd.DataFrame, dataset_2014: pd.DataFrame,
                                  dataset_2015: pd.DataFrame, dataset_2016: pd.DataFrame,
                                  variable_to_plot: str) -> None:
    """Create a grid of boxplots for monthly data of a variable across multiple years.

    Args:
        dataset_2013 (pd.DataFrame): DataFrame containing the time series data for the year 2013.
        dataset_2014 (pd.DataFrame): DataFrame containing the time series data for the year 2014.
        dataset_2015 (pd.DataFrame): DataFrame containing the time series data for the year 2015.
        dataset_2016 (pd.DataFrame): DataFrame containing the time series data for the year 2016.
        variable_to_plot (str): Name of the variable (column) to be plotted.

    Returns:
        None
    """
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 6), sharex=True, sharey=True, tight_layout=True)
    fig.suptitle(f'Box-plot "{variable_to_plot}" Monthly Data Comparison', weight='bold')

    b_2013 = sns.boxplot(
        x=dataset_2013['Month'], y=dataset_2013[variable_to_plot], hue=dataset_2013['Month'],
        ax=axes[0, 0], palette='husl', legend=False
    )
    b_2014 = sns.boxplot(
        x=dataset_2014['Month'], y=dataset_2014[variable_to_plot], hue=dataset_2014['Month'],
        ax=axes[0, 1], palette='husl', legend=False
    )
    b_2015 = sns.boxplot(
        x=dataset_2015['Month'], y=dataset_2015[variable_to_plot], hue=dataset_2015['Month'],
        ax=axes[1, 0], palette='husl', legend=False
    )
    b_2016 = sns.boxplot(
        x=dataset_2016['Month'], y=dataset_2016[variable_to_plot], hue=dataset_2016['Month'],
        ax=axes[1, 1], palette='husl', legend=False
    )

    b_2013 = b_2013.set_title('2013')
    b_2014 = b_2014.set_title('2014')
    b_2015 = b_2015.set_title('2015')
    b_2016 = b_2016.set_title('2016')

    axes[0, 0].set_xlabel('')
    axes[0, 1].set_xlabel('')
    plt.show()


def plot_variable_violinplot_monthly(dataset_2013: pd.DataFrame, dataset_2014: pd.DataFrame,
                                     dataset_2015: pd.DataFrame, dataset_2016: pd.DataFrame,
                                     variable_to_plot: str) -> None:
    """Create a grid of violin-plots for monthly data of a variable across multiple years.

    Args:
        dataset_2013 (pd.DataFrame): DataFrame containing the time series data for the year 2013.
        dataset_2014 (pd.DataFrame): DataFrame containing the time series data for the year 2014.
        dataset_2015 (pd.DataFrame): DataFrame containing the time series data for the year 2015.
        dataset_2016 (pd.DataFrame): DataFrame containing the time series data for the year 2016.
        variable_to_plot (str): Name of the variable (column) to be plotted.

    Returns:
        None
    """
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 6), sharex=True, sharey=True, tight_layout=True)
    fig.suptitle(f'Box-plot "{variable_to_plot}" Monthly Data Comparison', weight='bold')

    b_2013 = sns.violinplot(
        x=dataset_2013['Month'], y=dataset_2013[variable_to_plot], hue=dataset_2013['Month'],
        ax=axes[0, 0], palette='husl', legend=False
    )
    b_2014 = sns.violinplot(
        x=dataset_2014['Month'], y=dataset_2014[variable_to_plot], hue=dataset_2014['Month'],
        ax=axes[0, 1], palette='husl', legend=False
    )
    b_2015 = sns.violinplot(
        x=dataset_2015['Month'], y=dataset_2015[variable_to_plot], hue=dataset_2015['Month'],
        ax=axes[1, 0], palette='husl', legend=False
    )
    b_2016 = sns.violinplot(
        x=dataset_2016['Month'], y=dataset_2016[variable_to_plot], hue=dataset_2016['Month'],
        ax=axes[1, 1], palette='husl', legend=False
    )

    b_2013 = b_2013.set_title('2013')
    b_2014 = b_2014.set_title('2014')
    b_2015 = b_2015.set_title('2015')
    b_2016 = b_2016.set_title('2016')

    axes[0, 0].set_xlabel('')
    axes[0, 1].set_xlabel('')
    plt.show()


def plot_variable_calendar(dataset: pd.DataFrame, variable_to_plot: str) -> None:
    """Create a calendar heatmap for a variable in a DataFrame.

    **Note:** This function requires the `calplot` library to be installed.

    Args:
        dataset (pd.DataFrame): DataFrame containing the time series data.
        variable_to_plot (str): Name of the variable to be plotted.

    Returns:
        None
    """
    # Calendar plots for each year
    calplot.calplot(
        dataset[variable_to_plot],
        cmap='jet',
        yearlabel_kws={'fontname': 'sans-serif'},
        suptitle=f'"{variable_to_plot}" Variable')
    plt.show()


def plot_variable_eda(dataset: pd.DataFrame, variable_to_plot: str,
                      dataset_2013: pd.DataFrame, dataset_2014: pd.DataFrame,
                      dataset_2015: pd.DataFrame, dataset_2016: pd.DataFrame) -> None:
    """Generate a comprehensive set of exploratory data analysis (EDA) plots for a
    variable in a DataFrame to explore its distribution and trends across the years

    Plots Generated:

    1. Time Series Trend: Line plot showing the variable's trend over time.
    2. Distribution (Histogram): Histogram with a density distribution to visualize the
    variable's distribution.
    3. Distribution (Box Plot & Violin Plot): Combined boxplot and violinplot to analyze
    the variable's spread, outliers, and overall distribution.
    4. Box Plot & Violin Plot (Yearly): Boxplots and violinplots comparing the variable's
    distribution across the years.
    5 & 6. Box Plot & Violin Plot (Monthly): Boxplots comparing the variable's
    distribution across months and years.
    7. Calendar Heatmap: Heatmap visualization representing the variable's values for
    each day within a year. Requires `calplot` library.

    Args:
        dataset (pd.DataFrame): DataFrame containing the time series data.
        variable_to_plot (str): Name of the variable to be plotted.
        dataset_2013 (pd.DataFrame): DataFrame containing the time series data for
        the year 2013.
        dataset_2014 (pd.DataFrame): DataFrame containing the time series data for
        the year 2014.
        dataset_2015 (pd.DataFrame): DataFrame containing the time series data for
        the year 2015.
        dataset_2016 (pd.DataFrame): DataFrame containing the time series data for
        the year 2016.

    Returns:
        None
    """
    # trend plot
    plot_variable_trend(
        dataset=dataset, variable_to_plot=variable_to_plot,
    )

    # distribution plot (histogram)
    plot_variable_histogram(
        dataset=dataset, variable_to_plot=variable_to_plot,
    )

    # distribution boxplot and violinplot (overall)
    plot_variable_boxplot_violinplot(
        dataset=dataset, variable_to_plot=variable_to_plot
    )

    # Box plot / Violin plot - Yearly Comparison
    plot_variable_boxplot_violinplot_yearly(
        dataset=dataset, variable_to_plot=variable_to_plot
    )

    # Box plot / Violin plot - Monthly Comparison
    plot_variable_boxplot_monthly(
        dataset_2013=dataset_2013, dataset_2014=dataset_2014, dataset_2015=dataset_2015, dataset_2016=dataset_2016,
        variable_to_plot=variable_to_plot
    )

    plot_variable_violinplot_monthly(
        dataset_2013=dataset_2013, dataset_2014=dataset_2014, dataset_2015=dataset_2015, dataset_2016=dataset_2016,
        variable_to_plot=variable_to_plot
    )

    plot_variable_calendar(
        dataset=dataset, variable_to_plot=variable_to_plot
    )


def plot_amplitude_spectrum(variable_to_plot: str, frequency_vector: np.ndarray,
                            peaks: np.ndarray, xlim: float) -> None:
    """Plot a double-sided and zoomed amplitude spectrum of a signal using
    Fast Fourier Transform (FFT) results.

    Args:
        variable_to_plot (str): Name of the variable (signal) used for spectrum analysis.
        frequency_vector (np.ndarray): NumPy array containing frequency values.
        peaks (np.ndarray): NumPy array containing the corresponding FFT amplitude values.
        xlim (float): Upper limit for the zoomed spectrum x-axis.
    """
    # plotting
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), tight_layout=True)
    fig.suptitle(f'Amplitude Spectrum ["{variable_to_plot}"]', weight='bold')

    # Complete version (double sided)
    axes[0].set_title('Double Sided FFT')
    axes[0].stem(frequency_vector, peaks, 'b', basefmt="-b")
    axes[0].set_xlabel('Frequency')
    axes[0].set_ylabel('FFT Amplitude |X(freq)|')

    # Zoom version
    axes[1].set_title(f'Zoom (0, {xlim})')
    axes[1].stem(frequency_vector, peaks, 'b', basefmt="-b")
    axes[1].set_xlabel('Frequency')
    axes[1].set_xlim(0, xlim)
    plt.show()


def plot_periodgram(variable_to_plot: str, f_per_density, Pxx_per_density,
                    xlim: float) -> None:
    """Plot the power spectral density (PSD) of a signal using a periodogram.

    Args:
        variable_to_plot (str): Name of the variable (signal) used for spectrum analysis.
        f_per_density (_type_): NumPy array containing the frequency values.
        Pxx_per_density (_type_): NumPy array containing the corresponding PSD values.
        xlim (float): Upper limit for the zoomed spectrum x-axis.
    """
    # plot
    plt.figure(figsize=(10, 5), tight_layout=True)
    plt.plot(f_per_density, Pxx_per_density)
    plt.title(f'Periodogram - Density ["{variable_to_plot}" - Zoom({xlim})]', weight='bold')
    plt.ylabel('PSD')
    plt.xlim(0, xlim)
    plt.xlabel('Frequency')
    plt.show()


def plot_overall_train_test_preds(title: str, train_data, test_data,
                                  y_train, y_test, predictions, predictions_df,
                                  lower_ci, upper_ci):
    # Plotting predictions vs actual values (show also train data)
    plt.figure(figsize=(15,3), tight_layout=True)
    plt.title(f"{title}", weight="bold")

    plt.plot(train_data.index, y_train, label='train')
    plt.plot(test_data.index, y_test, 'k',label='test')
    plt.plot(test_data.index, predictions, 'r', label='predictions')

    plt.axvline(test_data.index[0], linestyle='dashed', color='k')
    plt.fill_between(test_data.index, predictions_df[lower_ci], predictions_df[upper_ci], alpha=.1, color='crimson',label='prediction int.')

    plt.xlabel('date')
    plt.ylabel('meantemp')
    plt.legend(loc='lower left')
    plt.show()


def plot_actual_vs_preds(test_data, y_test, predictions,
                         predictions_df, lower_ci, upper_ci):
    # Plotting predictions vs actual values (focus only on test data)
    plt.figure(figsize=(15,3), tight_layout=True)
    plt.title(f"Actual vs Prediction", weight="bold")

    plt.plot(test_data.index, y_test, 'ko-' ,label='test')
    plt.plot(test_data.index, predictions, 'ro-', label='predictions')

    plt.fill_between(test_data.index, predictions_df[lower_ci], predictions_df[upper_ci], alpha=.1, color='crimson',label='prediction int.')

    plt.xlabel('date')
    plt.ylabel('meantemp')
    plt.legend(loc='upper left')
    plt.show()


def plot_residual_analysis(train: bool, title: str, residuals, y_train, y_test):
    # Training Residuals
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15,5))

    if train:
        fig.suptitle(f'Residual Analysis - Training Dataset {title}', weight='bold')
    else:
        fig.suptitle(f'Residual Analysis - Test Dataset {title}', weight='bold')
    
    ax[0,0].plot(residuals)
    
    plot_acf(residuals, ax=ax[0,1])
    
    sns.histplot(residuals, kde=True, stat='density', ax=ax[0,2])
    
    sns.boxplot(x=residuals, ax=ax[1,0], showmeans=True)

    sm.qqplot(residuals, line='q', ax=ax[1,1])

    sns.residplot(y=residuals, x=y_train if train else y_test, lowess=True, ax=ax[1,2])
    plt.show()


def plot_overall_train_test_preds_no_ci(title: str, train_data, test_data,
                                  y_train, y_test, predictions):
    # Plotting predictions vs actual values (show also train data)
    plt.figure(figsize=(15,3), tight_layout=True)
    plt.title(f"{title}", weight="bold")

    plt.plot(train_data.index, y_train, label='train')
    plt.plot(test_data.index, y_test, 'k',label='test')
    plt.plot(test_data.index, predictions, 'r', label='predictions')

    plt.axvline(test_data.index[0], linestyle='dashed', color='k')

    plt.xlabel('date')
    plt.ylabel('meantemp')
    plt.legend(loc='lower left')
    plt.show()


def plot_actual_vs_preds_no_ci(test_data, y_test, predictions):
    # Plotting predictions vs actual values (focus only on test data)
    plt.figure(figsize=(15,3), tight_layout=True)
    plt.title(f"Actual vs Prediction", weight="bold")

    plt.plot(test_data.index, y_test, 'ko-' ,label='test')
    plt.plot(test_data.index, predictions, 'ro-', label='predictions')

    plt.xlabel('date')
    plt.ylabel('meantemp')
    plt.legend(loc='upper left')
    plt.show()

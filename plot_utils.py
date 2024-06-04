import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calplot


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

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_variable_trend(dataset: pd.DataFrame, variable_to_plot: str) -> None:
    """_summary_
    
    Args:
        dataset (pd.DataFrame): _description_
        variable_to_plot (str): _description_
    """
    plt.figure(figsize=(10, 3), tight_layout=True)
    plt.plot(dataset[variable_to_plot], label=variable_to_plot)
    plt.title(f'Time-Series of "{variable_to_plot}" Variable', weight='bold')
    plt.xlabel('Time [daily frequency]')
    plt.ylabel(f'{variable_to_plot}')
    plt.legend()
    plt.show()


def plot_variable_histogram(dataset: pd.DataFrame, variable_to_plot: str) -> None:
    """_summary_

    Args:
        dataset (pd.DataFrame): _description_
        variable_to_plot (str): _description_
    """
    plt.figure(figsize=(10, 3), tight_layout=True)
    plt.title(f'Histogram of "{variable_to_plot}" Variable', weight='bold')
    sns.histplot(data=dataset, x=variable_to_plot, kde=True, element='step')
    plt.show()


def plot_variable_boxplot_violinplot(dataset: pd.DataFrame, variable_to_plot: str) -> None:
    """_summary_

    Args:
        dataset (pd.DataFrame): _description_
        variable_to_plot (str): _description_
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
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=True, sharey=True, tight_layout=True)
    fig.suptitle(f'Box-plot and Violin-plot Yearly Data Comparison of "{variable_to_plot}" Variable', weight='bold')
    sns.boxplot(data=dataset, x=dataset['Year'], y=variable_to_plot, ax=ax[0])
    sns.violinplot(data=dataset, x=dataset['Year'], y=variable_to_plot, ax=ax[1])
    ax[0].set_xlabel('')
    plt.show()

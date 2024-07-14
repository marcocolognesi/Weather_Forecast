import numpy as np
import pandas as pd
from scipy import signal
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error


def find_IQR_outliers(dataset: pd.DataFrame, variable_to_analyze: str) -> list:
    """Identify outliers in a DataFrame column using the Interquartile Range (IQR) method.

    Steps:
    1. Find IQR (Q3 - Q1) to define thresholds for outlier detection.
    2. Values less than (Q1 - 1.5 * IQR) or greater than (Q3 + 1.5 * IQR) are considered outliers.

    The function also prints informative messages during execution.

    Args:
        dataset (pd.DataFrame): DataFrame containing the data.
        variable_to_analyze (str): Name of the variable (column) to analyze.

    Returns:
        list: List containing the identified outlier values from the specified variable.
    """
    # Finding Q1 (25% percentile) and Q3 (75% percentile)
    Q1 = np.quantile(dataset[variable_to_analyze], 0.25)
    Q3 = np.quantile(dataset[variable_to_analyze], 0.75)

    print(f"Q1 for variable variable is: {Q1}")
    print(f"Q3 for variable variable is: {Q3}")
    print("=================================")

    # Finding IQR -> Q3-Q1
    IQR = Q3 - Q1

    print(f"IQR for variable variable is: {IQR}")
    print("=================================")

    # Finding limits (minimum and maximum)
    lower_limit = Q1 - 1.5*IQR
    upper_limit = Q3 + 1.5*IQR

    print(f"Lower limit for outlier detection is: {lower_limit}")
    print(f"Upper limit for outlier detection is: {upper_limit}")
    print("=================================")

    # Finding outliers (if a value is < than lower limit or > than upper limit)
    outliers = [x for x in dataset[variable_to_analyze] if x < lower_limit or x > upper_limit]

    print(f"{len(outliers)} outliers were found:")
    return outliers


def update_year_dfs(total_dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame,
                                                          pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """_summary_

    Args:
        total_dataset (pd.DataFrame): _description_

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: _description_
    """
    # Update separate dataframes for each year
    # 2013
    df_2013 = total_dataset.loc[total_dataset.index.year == 2013]

    # 2014
    df_2014 = total_dataset.loc[total_dataset.index.year == 2014]

    # 2015
    df_2015 = total_dataset.loc[total_dataset.index.year == 2015]

    # 2016
    df_2016 = total_dataset.loc[total_dataset.index.year == 2016]
    return df_2013, df_2014, df_2015, df_2016


def find_frequency_peaks(Pxx_per_density, f_per_density) -> pd.DataFrame:
    # Finding frequency peaks
    peaks = signal.find_peaks(Pxx_per_density[f_per_density >= 0], prominence=6000)[0]
    peak_freq = f_per_density[peaks]
    peak_power = Pxx_per_density[peaks]

    # creating pd.DataFrame for better visual
    table = {'Freq': peak_freq, 'Period': 1/peak_freq, 'Power': peak_power}
    tab = pd.DataFrame(table)
    return tab


def compute_mae(y_test, preds):
    return mean_absolute_error(y_test, preds)


def compute_mse(y_test, preds):
    return mean_squared_error(y_test, preds)


def compute_rmse(y_test, preds):
    return root_mean_squared_error(y_test, preds)


def compute_r2(y_test, preds):
    return r2_score(y_test, preds) 


def compute_metrics(y_test, preds):
    mae = compute_mae(y_test, preds)
    mse = compute_mse(y_test, preds)
    rmse = compute_rmse(y_test, preds)
    r2 = compute_r2(y_test, preds)
    return mae, mse, rmse, r2
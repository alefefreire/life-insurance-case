import numpy as np
import pandas
from scipy import stats
from typing import Tuple


def detect_outliers(data: pandas.Series) -> dict:
    """
    This method detects outliers in a given data using
    the Inter Quartile Range (IQR) method.
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Outliers
    outliers = data[(data < lower_bound) | (data > upper_bound)]

    return {
        "outliers": outliers,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "percentage": len(outliers) / len(data) * 100,
    }


def spearman_corr(x: pandas.Series, y: pandas.Series, alpha: float = 0.05):
    """
    Performs Spearman rank correlation test between two variables.

    Parameters:
    -----------
    x : array-like
        First variable
    y : array-like
        Second variable
    alpha : float, optional (default=0.05)
        Significance level

    Returns:
    --------
    dict
        Dictionary containing:
        - correlation: Spearman correlation coefficient
        - pvalue: p-value of the test
        - significant: boolean indicating if correlation is significant
    """

    # Remove missing values
    mask = ~(np.isnan(x) | np.isnan(y))
    x = np.array(x)[mask]
    y = np.array(y)[mask]

    # Calculate Spearman correlation
    correlation, pvalue = stats.spearmanr(x, y)

    return {"correlation": correlation, "pvalue": pvalue, "significant": pvalue < alpha}


def kruskal_wallis_test(df: pandas.DataFrame, categorical_var: str, numeric_var: str) -> Tuple[float, float]:
    """
    Perform the Kruskal-Wallis test between a categorical variable and a numeric variable.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - categorical_var (str): The name of the categorical variable.
    - numeric_var (str): The name of the numeric variable.

    Returns:
    - tuple: A tuple containing the Kruskal-Wallis test statistic and the p-value.
    """
    # Prepare data for the Kruskal-Wallis test
    grouped_data = [df[numeric_var][df[categorical_var] == category] for category in df[categorical_var].unique()]

    # Apply the Kruskal-Wallis test
    statistic, p_value = stats.kruskal(*grouped_data)

    return statistic, p_value

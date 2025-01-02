import numpy as np
import pandas
from scipy import stats
from typing import Tuple
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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

    
class ModelEvaluator:
    def __init__(self, model: CatBoostRegressor, validation_df: pandas.DataFrame, target_column: str):
        """
        Initializes the model evaluator
        
        Parameters:
        -----------
        model : trained model (e.g., CatBoostRegressor)
            The trained model to be evaluated
        validation_df : pandas.DataFrame
            Validation DataFrame containing features and target
        target_column : str
            Name of the target column in the DataFrame
        """
        self.model = model
        self.validation_df = validation_df
        self.target_column = target_column
        
    def calculate_metrics(self):
        """
        Calculates all evaluation metrics
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing all evaluation metrics
        """
        # Split features and target
        X_val = self.validation_df.drop(columns=[self.target_column])
        y_val = self.validation_df[self.target_column]
        
        # Make predictions
        y_pred = self.model.predict(X_val)
        
        # Calculate metrics
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred)
        rmsle = np.sqrt(mean_squared_error(np.log1p(y_val), np.log1p(y_pred)))
        mape = np.mean(np.abs((y_val - y_pred) / y_val))
        
        # Create metrics DataFrame
        metrics_df = pandas.DataFrame({
            'Model': [self.model.__class__.__name__],
            'MAE': [mae],
            'MSE': [mse],
            'RMSE': [rmse],
            'R2': [r2],
            'RMSLE': [rmsle],
            'MAPE': [mape]
        })
        
        return metrics_df
    
    def evaluate_model(self, decimals=4):
        """
        Returns formatted metrics with specific number of decimal places
        
        Parameters:
        -----------
        decimals : int, optional (default=4)
            Number of decimal places to round to
            
        Returns:
        --------
        pandas.DataFrame
            Formatted DataFrame with metrics
        """
        metrics_df = self.calculate_metrics()
        
        # Format numeric columns
        numeric_columns = ['MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE']
        metrics_df[numeric_columns] = metrics_df[numeric_columns].round(decimals)
        
        return metrics_df
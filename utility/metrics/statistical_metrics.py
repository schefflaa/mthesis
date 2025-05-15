import warnings
import numpy as np
import pandas as pd
from typing import List


MINIMAL_STANDARD_DEVIATION_PRODUCT = 1e-16
DEFAULT_NUMBER_OF_INTERVALS: int = 100
DEFAULT_VIRTUAL_INDEX_METHOD: str = "median_unbiased"


def compute_statistics(df:pd.DataFrame):
    """
    Compute metrics for the model.

    Args:
    df: pd.DataFrame: DataFrame containing the predictions and the ground truth.
    - df["timestamp"]: Timestamp of the prediction.
    - df["y_true"]: Ground truth.
    - df["y_pred"]: Predictions.
    - df["batch"]: Batch number.
    """
    
    return {
        "bias":             bias(df["y_true"], df["y_pred"]),
        "association":      association(df["y_true"], df["y_pred"]),
        "discrimination":   discrimination(df["y_true"], df["y_pred"]),
        "underestimation":  underestimation(df["y_true"], df["y_pred"]),
        "overestimation":   overestimation(df["y_true"], df["y_pred"])
    }


class Discrimination:
    COMPATIBLE_VIRTUAL_INDEX_METHODS = [
        "median_unbiased",
        "hazen",
        "normal_unbiased",
        "linear",
        "weibull",
        "interpolated_inverted_cdf",
    ]

    def __init__(
        self,
        target: pd.Series,
        other: pd.Series,
        number_of_intervals: int = DEFAULT_NUMBER_OF_INTERVALS,
        virtual_index_method: str = DEFAULT_VIRTUAL_INDEX_METHOD,
    ):
        """Constructor of discrimination class

        Parameters
        ----------
        target: Timeseries of e.g. observations.
        other: Timeseries of e.g. predictions.
        number_of_intervals: Intervals, into which the timeseries should be divided.
        virtual_index_method: Method which should be used for the calculation of the quantile indices.
        """
        self._check_target_and_other(target, other)
        self._check_virtual_index_method(virtual_index_method)
        number_of_intervals = self._check_number_of_intervals(number_of_intervals, target)
        self._number_of_samples = len(target)
        quantile_index_borders = self._calculate_quantile_index_borders(
            number_of_intervals, self._number_of_samples, virtual_index_method
        )
        self._contingency_table = self._calculate_table(target, other, quantile_index_borders)

    @staticmethod
    def _check_target_and_other(target: pd.Series, other: pd.Series):
        length_target = len(target)
        length_other = len(other)
        if length_target != length_other:
            raise IndexError(f"Length of target={length_target} does not match length of other={length_other}")

    def _check_number_of_intervals(self, number_of_intervals: int, target: pd.Series) -> int:
        length_target = len(target)
        if number_of_intervals <= 0:
            raise ValueError(
                f"""{number_of_intervals=} is too small. number_of_intervals has to be greater than zero!"""
            )
        elif number_of_intervals > length_target:
            warnings.warn("number_of_intervals is greater than or equal to number of samples. It was set to number_of_samples")
        return min(number_of_intervals, length_target)

    @classmethod
    def _check_virtual_index_method(cls, virtual_index_method: str):
        if virtual_index_method not in cls.COMPATIBLE_VIRTUAL_INDEX_METHODS:
            raise ValueError(f"""{virtual_index_method=} is not a valid virtual index method!
                Available methods are {cls.COMPATIBLE_VIRTUAL_INDEX_METHODS}""")
        elif virtual_index_method == "interpolated_inverted_cdf":
            warnings.warn("""Passed virtual_index_method lead to inconsistent bin sizes during testing.
                If the number_of_intervals is close to or exactly the same as the length of the data frame some
                intervals may be empty.""")

    @staticmethod
    def _calculate_quantile_index_borders(number_of_intervals: int, number_of_samples: int, virtual_index_method: str) -> List[float]:
        quantile_names = np.linspace(0, 1, number_of_intervals + 1)
        quantile_index_borders = [calculate_virtual_index(number_of_samples, quantile, virtual_index_method) for quantile in quantile_names]
        return quantile_index_borders

    def _calculate_table(self, target: pd.Series, other: pd.Series, quantile_index_borders: List[float]) -> pd.DataFrame:
        temp_df = pd.DataFrame({"target": target, "other": other})
        temp_df.sort_values(by=["target", "other"], inplace=True)
        temp_df.reset_index(inplace=True, drop=True)
        temp_df["interval_number_target"] = pd.cut(temp_df.index, list(quantile_index_borders), labels=False)

        temp_df.sort_values(by=["other", "target"], inplace=True)
        temp_df.reset_index(inplace=True, drop=True)
        temp_df["interval_number_other"] = pd.cut(temp_df.index, list(quantile_index_borders), labels=False)

        interval_counts = temp_df.groupby(["interval_number_other", "interval_number_target"]).size()
        contingency_table = interval_counts.unstack().fillna(0)
        return contingency_table

    @property
    def contingency_table(self) -> pd.DataFrame:
        return self._contingency_table

    @property
    def score(self) -> float:
        return self.contingency_table.values.trace() / self._number_of_samples

    @property
    def anticorrelation_score(self) -> float:
        return np.flip(self._contingency_table.values, axis=1).trace() / self._number_of_samples

def calculate_virtual_index(number_of_samples, quantile_name, method="median_unbiased"):
    """

    the quantiles are added based on the index
    index calculation is based on the formula described by [1]
    the index i for a quantile q will be calculated with the formula:
        i + g = q*(n - alpha - beta + 1) + alpha
    because pythons indexing starts with zero, one will be subtracted from the index
        i + g = q*(n - alpha - beta + 1) + alpha -1

    Calculated indices are between -1 and number_of_samples.


    For unknown distributions the method "median_unbiased" is recommended.


    Sources:
    [1] https://numpy.org/doc/stable/reference/generated/numpy.quantile.html#id1, 2023-02-14
    [2] https://de.wikipedia.org/wiki/Empirisches_Quantil, 2023-02-14
    [3] R. J. Hyndman and Y. Fan, “Sample quantiles in statistical packages,”
        The American Statistician, 50(4), pp. 361-365, 1996
    """
    if method == "median_unbiased":
        alpha = 1 / 3
        beta = 1 / 3
    elif method == "hazen":
        alpha = 1 / 2
        beta = 1 / 2
    elif method == "normal_unbiased":
        alpha = 3 / 8
        beta = 3 / 8
    elif method == "linear":
        alpha = 1
        beta = 1
    elif method == "weibull":
        alpha = 0
        beta = 0
    elif method == "interpolated_inverted_cdf":
        alpha = 0
        beta = 1
    else:
        warnings.warn(
            """Method for virtual index calculation of quantiles has not been defined!
            It was automatically set to 'weibull'"""
        )
        alpha = 0
        beta = 0

    return quantile_name * (number_of_samples - alpha - beta + 1) + alpha - 1


def accuracy(target: pd.Series, other: pd.Series, normalization: int = 1) -> float:
    """Mean Absolute Error (mae)

    Parameters
    ----------
    target
        Timeseries of e.g. observations.
    other
        Timeseries of e.g. predictions.
    normalization
        normalize by this value

    Returns
    -------
    mean absolute error as float.


    Remarks
    -------
    Formula:

        mae(target, other) = 1/N sum( |target_i - other_i|)

    with N = sample size, target_i = target at index i, other_i = other at index i
    Further information can be found in the demonstration/ directory in the corresponding notebook
    """

    return (target - other).abs().mean() / normalization


def association(target: pd.Series, other: pd.Series) -> float:
    """Correlation between two time series

    Parameters
    ----------
    target
        Timeseries of e.g. observations. pd.Series.dtype should not be object! Use float or int instead.
    other
        Timeseries of e.g. predictions. pd.Series.dtype should not be object! Use float or int instead.

    Returns
    -------
    correlation

    Remarks
    -------
    Formula:

        corr(target, other) = covariance(target, other) / (standard_deviation(target) * standard_deviation(other))

    Further information can be found in the demonstration/ directory in the corresponding notebook
    """
    product_of_standard_deviations = target.std() * other.std()
    if product_of_standard_deviations <= MINIMAL_STANDARD_DEVIATION_PRODUCT:
        return float("nan")
    return other.cov(target) / product_of_standard_deviations


def bias(target: pd.Series, other: pd.Series, normalization: int = 1) -> float:
    """Bias

    Parameters
    ----------
    target
        Timeseries of e.g. observations.
    other
        Timeseries of e.g. predictions.
    normalization
        normalize by this value

    Returns
    -------
    bias

    Remarks
    -------
    Formula:

        bias(target, other) = 1/N sum( target_i - other_i)

    with N = sample size, target_i = target at index i, other_i = other at index i
    Further information can be found in the demonstration/ directory in the corresponding notebook
    """
    return (other - target).mean() / normalization


def root_mean_square_error(target: pd.Series, other: pd.Series, normalization: int = 1) -> float:
    """Root Mean Square Error (rmse)

    Parameters
    ----------
    target
        Timeseries of e.g. observations.
    other
        Timeseries of e.g. predictions.
    normalization
        normalize by this value

    Returns
    -------
    root mean square error

    Remarks
    -------
    Formula:

        rmse(target, other) = sqrt( 1/N sum( (target_i - other_i)² ) )

    with N = sample size, target_i = target at index i, other_i = other at index i
    Further information can be found in the demonstration/ directory in the corresponding notebook
    """
    return np.sqrt((target - other).pow(2).mean()) / normalization


def discrimination(
    target: pd.Series,
    other: pd.Series,
    number_of_intervals: int = 100,
    virtual_index_method: str = "median_unbiased",
) -> float:
    """Discrimination score for data with positive relation (e.g. positively correlated data)

    Parameters
    ----------
    target
        Timeseries of e.g. observations.
    other
        Timeseries of e.g. predictions.
    number_of_intervals
        Intervals, into which the timeseries should be divided. If None, the default value
        NUMBER_OF_INTERVALS from the Discrimination class will be used.
    virtual_index_method
        Method which should be used for the calculation of the quantile indices. If None, the default
        value VIRTUAL_INDEX_METHOD from the Discrimination class will be used.

    Returns
    -------
    discrimination score

    Remarks
    -------
    Algorithm:

        0. Calculate index borders for the intervals.
        1. Align target and other are in one df.
        2. Sort df after other.
        3. Sort df after target, keeping the order of other for equal target values.
        4. Add a column to the df, which remembers which target value belongs to which interval.
        5. Sort df after other, keeping the order of target for equal other values.
        6. Add a column to the df, which remembers which other value belongs to which interval.
        7. Calculate the number n_diagonal of data points which landed in the same target and other interval.
        8. Calculate the ratio of n_diagonal to the total number of data points.

    """
    return Discrimination(target, other, number_of_intervals, virtual_index_method).score


def underestimation(
    target: pd.Series,
    other: pd.Series,
) -> float:
    """
    Underestimation normalized with the sum over whole target

    Parameters
    ----------
    target
        Timeseries of e.g. observations.
    other
        Timeseries of e.g. predictions.

    Returns
    -------
    underestimation: float
    """

    residual = other - target
    abs_residual = residual.abs()
    if np.nansum(target) == 0:
        warnings.warn("Division by zero: Target series is constant zero, setting underestimation to nan")
        return np.nan
    else:
        return np.nansum(np.where(residual < 0, abs_residual, 0)) / np.nansum(target)


def overestimation(
    target: pd.Series,
    other: pd.Series,
) -> float:
    """
    Overestimation normalized with the sum over whole target

    Parameters
    ----------
    target
        Timeseries of e.g. observations.
    other
        Timeseries of e.g. predictions.

    Returns
    -------
    overestimation: float
    """
    residual = other - target
    abs_residual = residual.abs()
    if np.nansum(target) == 0:
        warnings.warn("Division by zero: Target series is constant zero, setting overestimation to nan.")
        return np.nan
    else:
        return np.nansum(
            np.where(residual > 0, abs_residual, 0),
        ) / np.nansum(target)

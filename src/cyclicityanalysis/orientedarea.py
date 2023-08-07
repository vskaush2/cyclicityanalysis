"""
orientedarea.py
~~~~~~~~~~~~~~~~
This module contains all necessary oriented area calculations for a multivariate time-series
"""

import pandas as pd
import numpy as np


class OrientedArea:
    """A class containing different oriented area calculations
        for a given multivariate time-series

        Attributes:
            df (pd.DataFrame): The given multivariate time-series
            columns (pd.Index): The column names of `df`
            index (pd.Index): The index of `df`
            N (int): The number of columns of `df`

        """

    def __init__(self, df: pd.DataFrame | np.ndarray):
        """

        Args:
            df: The given multivariate time-series
        Raises:
            ValueError: If one of the following situations occurs:
                `df` is neither a pandas dataframe nor numpy array.
                `df` has either only one row or only one column, or
                `df` has missing values.

        """
        if isinstance(df, pd.DataFrame):
            self.df = df
        elif isinstance(df, np.ndarray):
            self.df = pd.DataFrame(df, columns=[str(x) for x in df.shape[-1]])
        else:
            raise ValueError("The given time-series must either be a pandas dataframe or a numpy array !")
        if self.df.shape[0] <= 1:
            raise ValueError("The given time-series must have more than one row !")
        if self.df.shape[-1] <= 1:
            raise ValueError("The given time-series must have more than one column !")
        if self.df.isnull().values.any():
            raise ValueError("The given time-Series must have no missing values !")

        self.columns = self.df.columns
        self.index = self.df.index
        self.N = self.df.shape[-1]
        self.df_diff = self.df.diff()

    def compute_pairwise_oriented_area(self, col1: str, col2: str, start_time=None, end_time=None) -> float:
        """Computes the pairwise oriented area for two columns of df
            over a time period

        Args:
            col1: First column name of `df`
            col2: Second column name of `df`
            start_time: Starting index for the oriented area calculation. Default is None.
                If None, `start_time` is set to the index corresponding to the first row of `df`
            end_time: Ending index for the oriented area calculation. Default is None.
                If None, `end_time` is set to the index corresponding to the last row of `df`

        Returns:
            float: The pairwise oriented area
        """
        if col1 == col2:
            return 0
        if start_time is None:
            start_time = self.index[0]
        if end_time is None:
            end_time = self.index[-1]
        X = self.df[[col1, col2]].loc[start_time: end_time]
        dX = self.df_diff[[col1, col2]].loc[start_time: end_time]
        x, y = X[col1].values[:-1], X[col2].values[:-1]
        dx, dy = dX[col1].values[1:], dX[col2].values[1:]
        pairwise_oriented_area = 0.5 * (x @ dy - y @ dx)
        return pairwise_oriented_area

    def compute_pairwise_accumulated_oriented_area_df(self, col1: str, col2: str, start_time=None,
                                                      end_time=None) -> pd.DataFrame:
        """Computes the pairwise accumulated oriented area time-series
            for two columns of `df` over a time period

        Args:
            col1: First column name of `df`
            col2: Second column name of `df`
            start_time: Starting index for the oriented area calculation. Default is None.
                If None, `start_time` is set to the index corresponding to the first row of `df`
            end_time: Ending index for the oriented area calculation. Default is None.
                If None, `end_time` is set to the index corresponding to the last row of `df`

        Returns:
            pd.DataFrame: The pairwise accumulated oriented area time-series
        """
        if start_time is None:
            start_time = self.index[0]
        if end_time is None:
            end_time = self.index[-1]
        column = "{} - {} Accumulated Oriented Area".format(col1, col2)
        index = self.df.loc[start_time: end_time].index
        pairwise_accumulated_oriented_area_df = pd.DataFrame(columns=[column], index=index)
        if col1 != col2:
            dX = self.df_diff[[col1, col2]].loc[start_time: end_time]
            change_times = dX[(dX[col1] != 0) | (dX[col2] != 0)].index
            pairwise_accumulated_oriented_area_df.loc[change_times] \
                = np.array([self.compute_pairwise_oriented_area(col1, col2, start_time, change_time)
                            for change_time in change_times]).reshape(-1, 1)
            pairwise_accumulated_oriented_area_df.fillna(method='ffill', inplace=True)

        pairwise_accumulated_oriented_area_df.fillna(0, inplace=True)
        return pairwise_accumulated_oriented_area_df

    def compute_lead_lag_df(self, start_time=None, end_time=None) -> pd.DataFrame:
        """Computes the lead lag matrix of `df` over a time period

        Args:
            start_time: Starting index for the lead lag matrix calculation. Default is None.
                If None, `start_time` is set to the index corresponding to the first row of `df`
            end_time: Ending index for the lead lag matrix calculation. Default is None.
                If None, `end_time` is set to the index corresponding to the last row of `df`

        Returns:
            pd.DataFrame: The lead lag matrix
        """
        if start_time is None:
            start_time = self.index[0]
        if end_time is None:
            end_time = self.index[-1]
        X = self.df.loc[start_time: end_time]
        dX = self.df_diff.loc[start_time: end_time]
        A = X.values[:-1].T @ dX.values[1:]
        Q = 0.5 * (A - A.T)
        lead_lag_df = pd.DataFrame(Q, columns=self.columns, index=self.columns)
        return lead_lag_df

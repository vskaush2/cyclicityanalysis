"""
coom.py
~~~~~~~~~~~~~~~~
This module fits the Chain of Offsets Model (COOM) on a skew-symmetric matrix
"""

import pandas as pd
import numpy as np


class COOM:
    """A class fitting Chain of Offsets Model (COOM) on a given skew-symmetric matrix
  
      Attributes:
        N (int): The number of columns of `skew_symmetric_df`
        skew_symmetric_df (pd.DataFrame): The given skew-symmetric matrix
        columns (pd.Index): The column names of `skew_symmetric_df`
        eigenvalues (np.ndarray): The eigenvalues of `skew_symmetric_df`
        eigenvectors (np.ndarray): The matrix of eigenvectors of `skew_symmetric_df`,
            where the nth column is the eigenvector corresponding to the nth member of `eigenvalues`
        eigenvalue_moduli (np.ndarray): The moduli of all members in `eigenvalues`
        sorted_eigenvalue_indices (np.ndarray): Sorted indices between 0 and `N`-1 (inclusive)
            such that the n-th member of the list is the nth largest member of `eigenvalue_moduli`
  """

    def __init__(self, skew_symmetric_df: pd.DataFrame | np.ndarray):
        """

      Args:
        skew_symmetric_df: The given skew-symmetric matrix

      Raises:
        ValueError: If one of the following situations occurs:
          `skew_symmetric_df` is neither a pandas dataframe nor numpy array
          `skew_symmetric_df` is not a square matrix
          `skew_symmetric_df` does not have more than one entry
          `skew_symmetric_df` is not a skew-symmetric matrix

      """
        if not isinstance(skew_symmetric_df, (pd.DataFrame, np.ndarray)):
            raise ValueError("Lead-Lag Matrix must either be a Pandas DataFrame or Numpy Array")
        if skew_symmetric_df.shape[0] != skew_symmetric_df.shape[-1]:
            raise ValueError("Lead-Lag Matrix must be a square matrix")
        if skew_symmetric_df.shape[0] <= 1:
            raise ValueError("Lead-Lag Matrix must have more than one entry")
        if not np.allclose(skew_symmetric_df.T, -skew_symmetric_df):
            raise ValueError("Lead-Lag Matrix must be a skew-symmetric matrix")

        self.N = skew_symmetric_df.shape[0]

        if isinstance(skew_symmetric_df, np.ndarray):
            skew_symmetric_df = pd.DataFrame(skew_symmetric_df, columns=[str(x) for x in range(self.N)])

        self.skew_symmetric_df = skew_symmetric_df
        self.columns = self.skew_symmetric_df.columns
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.skew_symmetric_df)
        self.eigenvalue_moduli = np.abs(self.eigenvalues)
        self.sorted_eigenvalue_indices = np.argsort(self.eigenvalue_moduli)[::-1]

    def get_leading_eigenvector(self, n: int = 0) -> np.ndarray:
        """Gets the eigenvector corresponding to (N-n)-th largest eigenvalue (in modulus)
        of `skew_symmetric_df`

        Args:
            n: An index between 0 and `N`-1 (inclusive). Default is 0
        Returns:
            np.ndarray: The eigenvector corresponding to the (`N`-`n`)-th largest eigenvalue

        """
        leading_eigenvector = self.eigenvectors[:, self.sorted_eigenvalue_indices[n]]
        return leading_eigenvector

    def compute_phases(self, v: np.ndarray) -> np.ndarray:
        """Gets the phases of a given vector, which are the principal arguments of the vector's components

        Args:
            v: A vector with complex components

        Returns:
            np.ndarray: The phases

        """
        phases = np.angle(v)
        phases = np.array([-phase if phase >= np.pi else phase for phase in phases])
        return phases

    def compute_sequential_order_dict(self, v: np.ndarray) -> dict:
        """Computes the sequential order dictionary of a given vector according to COOM,
        in which we sort the components of the vector by their phases in increasing order;
        keys are the indices between 0 and `N`-1 sorted according to COOM
        values are corresponding column names of `skew_symmetric_df` sorted according to COOM


        Args:
            v: A vector with complex components

        Returns:
            dict:  Sequential order dictionary
        """
        phases = self.compute_phases(v)
        sequential_order_indices = np.argsort(phases)
        sequential_order_columns = [self.columns[index] for index in sequential_order_indices]
        sequential_order_dict = dict(zip(sequential_order_indices, sequential_order_columns))
        return sequential_order_dict

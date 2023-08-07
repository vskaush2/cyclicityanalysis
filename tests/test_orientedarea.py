from cyclicityanalysis.orientedarea import *
import pytest


@pytest.mark.parametrize(
    "N, K",
    [
        (2, 5),
        (3, 500),
        (4, 1000),
        (5, 1000),
        (6, 5000),
        (20, 10000),
        (100, 100000)
    ]
)
def test_compute_lead_lag_df(N, K):
    """Tests computation of the N x N lead lag matrix for
    multivariate time-series having (n,k)-th observation
    sin(2 pi (k-1)/K - pi n/N) for 1 <= k <= K+1 and 1 <= n <= N

    Parameters
    ----------
    N : int
        Number of Component Time-Series
    K : int
        Number of Observations

    Returns
    -------

    """
    t = np.linspace(0, 1, K + 1)
    sinusoidal_df = pd.DataFrame(np.array([np.sin(2 * np.pi * (t - n / N)) for n in range(N)]).T,
                                 columns=['$x_{{{}}}$'.format(n + 1) for n in range(N)])

    oa = OrientedArea(sinusoidal_df)
    computed_lead_lag_df = oa.compute_lead_lag_df().values.astype(float)
    correct_lead_lag_df = np.array(
        [[K/ 2 * np.sin(2 * np.pi / (K)) * np.sin(2*np.pi * (n - m) / N) for n in range(N)]
         for m in range(N)])
    assert np.allclose(computed_lead_lag_df, correct_lead_lag_df), "Lead Lag Matrices Do Not Agree !"

import numpy as np
from information_diffusion_model import (
    NetworkInformationDiffusion,
    KyleKMicrostructure,
    InformationDiffusionMarketModel,
    PINEstimator,
)


def test_diffusion_curve_monotonic_and_bounded():
    diffusion = NetworkInformationDiffusion(n_traders=200, n_seed=2, beta=0.08, seed=1)
    lam = diffusion.simulate(n_periods=40)
    assert (lam.diff().dropna() >= -1e-9).all(), "SI diffusion must be non-decreasing"
    assert lam.iloc[0] > 0 and lam.iloc[-1] <= 1.0


def test_market_simulation_shapes():
    diffusion = NetworkInformationDiffusion(n_traders=200, n_seed=2, beta=0.08, seed=1)
    micro = KyleKMicrostructure(n_traders=200, seed=2)
    model = InformationDiffusionMarketModel(diffusion, micro)
    df = model.simulate(n_periods=30, p0=100.0)
    assert len(df) == 31
    assert {"price", "buys", "sells", "lambda_t"}.issubset(df.columns)
    assert (df["buys"] >= 0).all() and (df["sells"] >= 0).all()


def test_pin_recovers_higher_informedness_when_more_informed():
    """A trade tape generated with a higher, constant informed fraction should
    yield a higher estimated PIN than one generated with a low fraction."""
    diffusion = NetworkInformationDiffusion(n_traders=300, n_seed=150, beta=0.5, seed=3)
    micro = KyleKMicrostructure(n_traders=300, seed=4)
    model = InformationDiffusionMarketModel(diffusion, micro)
    df_high = model.simulate(n_periods=5, p0=100.0)  # seeds most of the pop -> high lambda fast

    diffusion_low = NetworkInformationDiffusion(n_traders=300, n_seed=1, beta=0.01, seed=3)
    model_low = InformationDiffusionMarketModel(diffusion_low, micro)
    df_low = model_low.simulate(n_periods=5, p0=100.0)

    pin_high = PINEstimator().fit(df_high["buys"].values, df_high["sells"].values, n_restarts=4, seed=5)["PIN"]
    pin_low = PINEstimator().fit(df_low["buys"].values, df_low["sells"].values, n_restarts=4, seed=5)["PIN"]

    assert pin_high >= pin_low

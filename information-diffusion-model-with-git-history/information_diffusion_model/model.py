"""
Information Diffusion Model for Market Microstructure
=======================================================

Models how private information about an asset's value spreads through a
population of traders, and how that diffusion maps into observable market
quantities: order flow imbalance, price impact, and the probability of
informed trading (PIN).

Two layers, designed to be used independently or together:

1. NetworkInformationDiffusion
   An SI (Susceptible-Infected) epidemic process on a trader network.
   A signal originates at a small set of "informed" nodes and spreads
   through the network over discrete time steps. Produces a diffusion
   curve lambda(t) = fraction of the trading population that has the
   information at time t. This is the standard reduced-form way to
   capture "gradual information diffusion" effects (e.g. lead-lag /
   cross-asset momentum spillover, analyst coverage diffusion, etc.)

2. KyleKMicrostructure
   Given a time-varying informed fraction, generates order flow from
   informed and noise/uninformed traders each period, and computes the
   resulting price impact using a Kyle (1985) style linear pricing rule
   where the market maker's lambda (price sensitivity to order flow)
   adapts to the current proportion of informed trading.

3. PINEstimator
   Given observed buy/sell order flow, estimates the Easley-O'Hara (1992)
   Probability of INformed trading via maximum likelihood -- i.e. the
   inverse problem: recovering how "informed" the market was from trades
   alone. Useful for validating the simulation and for use on real data.

4. InformationDiffusionMarketModel
   Wires 1 + 2 + 3 together into a single simulate() -> analyze() pipeline.

No external network calls. Pure numpy/scipy/pandas/networkx/matplotlib.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
from dataclasses import dataclass, field
from scipy.optimize import minimize
from scipy.stats import poisson


# ----------------------------------------------------------------------
# 1. Network information diffusion (SI epidemic on a trader graph)
# ----------------------------------------------------------------------

@dataclass
class NetworkInformationDiffusion:
    """
    SI diffusion of a single piece of information across a trader network.

    n_traders     : size of the trader population
    topology      : "erdos_renyi", "watts_strogatz", or "barabasi_albert"
    topo_params   : dict of extra params for the chosen topology
    n_seed        : number of initially-informed traders (the "leak")
    beta          : per-edge, per-period transmission probability
    seed          : RNG seed for reproducibility
    """
    n_traders: int = 500
    topology: str = "watts_strogatz"
    topo_params: dict = field(default_factory=lambda: {"k": 8, "p": 0.1})
    n_seed: int = 3
    beta: float = 0.05
    seed: int = 42

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.graph = self._build_graph()

    def _build_graph(self) -> nx.Graph:
        if self.topology == "erdos_renyi":
            p = self.topo_params.get("p", 0.01)
            g = nx.erdos_renyi_graph(self.n_traders, p, seed=self.seed)
        elif self.topology == "watts_strogatz":
            k = self.topo_params.get("k", 8)
            p = self.topo_params.get("p", 0.1)
            g = nx.watts_strogatz_graph(self.n_traders, k, p, seed=self.seed)
        elif self.topology == "barabasi_albert":
            m = self.topo_params.get("m", 4)
            g = nx.barabasi_albert_graph(self.n_traders, m, seed=self.seed)
        else:
            raise ValueError(f"Unknown topology: {self.topology}")
        return g

    def simulate(self, n_periods: int = 60) -> pd.Series:
        """
        Run the SI process for n_periods discrete steps.
        Returns lambda(t): fraction of traders informed at each t (t=0..n_periods).
        """
        n = self.n_traders
        informed = np.zeros(n, dtype=bool)
        seed_nodes = self.rng.choice(n, size=self.n_seed, replace=False)
        informed[seed_nodes] = True

        adj = nx.to_scipy_sparse_array(self.graph, format="csr")
        frac_informed = np.zeros(n_periods + 1)
        frac_informed[0] = informed.mean()

        for t in range(1, n_periods + 1):
            # number of informed neighbors for every node
            informed_neighbor_count = adj.dot(informed.astype(float))
            # probability a susceptible node becomes informed this period:
            # 1 - (1-beta)^(#informed neighbors)  -- independent-contact SI update
            infect_prob = 1.0 - (1.0 - self.beta) ** informed_neighbor_count
            draws = self.rng.random(n)
            newly_informed = (~informed) & (draws < infect_prob)
            informed = informed | newly_informed
            frac_informed[t] = informed.mean()

        return pd.Series(frac_informed, index=np.arange(n_periods + 1), name="lambda_t")


# ----------------------------------------------------------------------
# 2. Kyle-style microstructure layer driven by the diffusion curve
# ----------------------------------------------------------------------

@dataclass
class KyleKMicrostructure:
    """
    Period-by-period Kyle(1985)-flavored pricing.

    fundamental_value : true (unobserved) asset value that informed traders know
    sigma_u           : std dev of per-period noise-trader order flow
    order_size_unit   : size of one informed trader's order when active
    n_traders         : must match the diffusion model's n_traders (for scaling flow)
    seed              : RNG seed
    """
    fundamental_value: float = 100.0
    sigma_u: float = 50.0
    order_size_unit: float = 1.0
    n_traders: int = 500
    seed: int = 7

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def _kyle_lambda(self, informed_frac: float) -> float:
        """
        Standard single-auction Kyle lambda, adapted to a time-varying
        informed fraction: lambda_t = sigma_v(informed_frac) / (2*sigma_u)
        where sigma_v scales with how much *aggregate* informed flow could
        move price this period (more informed traders -> more concentrated,
        predictable flow -> market maker sets a steeper price schedule).
        """
        sigma_v_effective = informed_frac * self.n_traders * self.order_size_unit
        lam = sigma_v_effective / (2.0 * self.sigma_u + 1e-9)
        return lam

    def simulate_prices(self, lambda_path: pd.Series, p0: float = 100.0) -> pd.DataFrame:
        """
        Given lambda_t (fraction informed each period), simulate:
          - informed order flow (net buy pressure toward fundamental_value)
          - noise order flow
          - total order flow
          - price update via price_t = price_{t-1} + kyle_lambda_t * order_flow_t
        """
        rows = []
        price = p0
        for t, frac in lambda_path.items():
            n_informed = int(round(frac * self.n_traders))
            direction = np.sign(self.fundamental_value - price) or 1.0
            informed_flow = n_informed * self.order_size_unit * direction
            noise_flow = self.rng.normal(0, self.sigma_u)
            total_flow = informed_flow + noise_flow

            lam = self._kyle_lambda(frac)
            price = price + lam * total_flow

            # microstructure-observable buy/sell trade counts (Poisson,
            # split by informed/uninformed, needed later for PIN estimation)
            buy_arrival_informed = n_informed if direction > 0 else 0
            sell_arrival_informed = n_informed if direction < 0 else 0
            noise_intensity = max(self.sigma_u / 5.0, 1.0)
            buys = poisson.rvs(noise_intensity, random_state=self.rng) + buy_arrival_informed
            sells = poisson.rvs(noise_intensity, random_state=self.rng) + sell_arrival_informed

            rows.append({
                "t": t,
                "lambda_t": frac,
                "kyle_lambda": lam,
                "informed_flow": informed_flow,
                "noise_flow": noise_flow,
                "total_flow": total_flow,
                "price": price,
                "buys": buys,
                "sells": sells,
            })
        return pd.DataFrame(rows).set_index("t")


# ----------------------------------------------------------------------
# 3. PIN estimator (Easley & O'Hara, 1992) -- recovers informedness from
#    observed buy/sell trade counts alone (the inverse problem)
# ----------------------------------------------------------------------

class PINEstimator:
    """
    Maximum-likelihood estimation of the Easley-O'Hara PIN model.

    Parameters estimated: alpha (prob. an information event occurs that day),
    delta (prob. the event is bad news | event occurred), mu (informed trade
    intensity), epsilon_b, epsilon_s (uninformed buy/sell intensities).

    PIN = (alpha * mu) / (alpha * mu + epsilon_b + epsilon_s)
    """

    @staticmethod
    def _neg_log_likelihood(params, buys: np.ndarray, sells: np.ndarray) -> float:
        alpha, delta, mu, eps_b, eps_s = params
        if not (0 < alpha < 1 and 0 < delta < 1 and mu > 0 and eps_b > 0 and eps_s > 0):
            return 1e10

        ll = 0.0
        for b, s in zip(buys, sells):
            # no event
            p_no_event = (1 - alpha) * poisson.pmf(b, eps_b) * poisson.pmf(s, eps_s)
            # bad news event (informed sell)
            p_bad = alpha * delta * poisson.pmf(b, eps_b) * poisson.pmf(s, eps_s + mu)
            # good news event (informed buy)
            p_good = alpha * (1 - delta) * poisson.pmf(b, eps_b + mu) * poisson.pmf(s, eps_s)
            day_lik = p_no_event + p_bad + p_good
            ll += np.log(max(day_lik, 1e-300))
        return -ll

    def fit(self, buys: np.ndarray, sells: np.ndarray, n_restarts: int = 8, seed: int = 0):
        rng = np.random.default_rng(seed)
        best = None
        b_mean, s_mean = np.mean(buys), np.mean(sells)

        for _ in range(n_restarts):
            x0 = np.array([
                rng.uniform(0.05, 0.5),                 # alpha
                rng.uniform(0.2, 0.8),                   # delta
                rng.uniform(1, max(b_mean, s_mean, 2)),  # mu
                rng.uniform(0.5, max(b_mean, 1)),        # eps_b
                rng.uniform(0.5, max(s_mean, 1)),        # eps_s
            ])
            res = minimize(
                self._neg_log_likelihood, x0, args=(buys, sells),
                method="Nelder-Mead",
                options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6},
            )
            if best is None or res.fun < best.fun:
                best = res

        alpha, delta, mu, eps_b, eps_s = best.x
        pin = (alpha * mu) / (alpha * mu + eps_b + eps_s)
        return {
            "alpha": alpha, "delta": delta, "mu": mu,
            "eps_b": eps_b, "eps_s": eps_s, "PIN": pin,
            "neg_log_lik": best.fun,
        }


# ----------------------------------------------------------------------
# 4. Full pipeline
# ----------------------------------------------------------------------

class InformationDiffusionMarketModel:
    """
    End-to-end: network diffusion -> microstructure price formation ->
    (optional) PIN re-estimation from the simulated trade tape, to check
    that the recovered informedness lines up with the diffusion curve
    that generated it.
    """

    def __init__(self, diffusion: NetworkInformationDiffusion, micro: KyleKMicrostructure):
        self.diffusion = diffusion
        self.micro = micro
        self.lambda_path_ = None
        self.market_df_ = None
        self.pin_result_ = None

    def simulate(self, n_periods: int = 60, p0: float = 100.0) -> pd.DataFrame:
        self.lambda_path_ = self.diffusion.simulate(n_periods=n_periods)
        self.market_df_ = self.micro.simulate_prices(self.lambda_path_, p0=p0)
        return self.market_df_

    def estimate_pin(self, window: int = None) -> dict:
        if self.market_df_ is None:
            raise RuntimeError("Call simulate() first.")
        df = self.market_df_ if window is None else self.market_df_.tail(window)
        est = PINEstimator().fit(df["buys"].values, df["sells"].values)
        self.pin_result_ = est
        return est

    def summary(self) -> pd.DataFrame:
        if self.market_df_ is None:
            raise RuntimeError("Call simulate() first.")
        df = self.market_df_.copy()
        df["order_flow_imbalance"] = (df["buys"] - df["sells"]) / (df["buys"] + df["sells"])
        return df

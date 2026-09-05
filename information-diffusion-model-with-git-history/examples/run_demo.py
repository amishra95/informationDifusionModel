import os
import numpy as np
import matplotlib.pyplot as plt

from information_diffusion_model import (
    NetworkInformationDiffusion,
    KyleKMicrostructure,
    InformationDiffusionMarketModel,
    PINEstimator,
)

np.random.seed(0)

diffusion = NetworkInformationDiffusion(
    n_traders=500,
    topology="watts_strogatz",
    topo_params={"k": 8, "p": 0.1},
    n_seed=3,
    beta=0.06,
    seed=42,
)

micro = KyleKMicrostructure(
    fundamental_value=105.0,
    sigma_u=40.0,
    order_size_unit=0.05,
    n_traders=500,
    seed=7,
)

model = InformationDiffusionMarketModel(diffusion, micro)
market_df = model.simulate(n_periods=60, p0=100.0)
summary = model.summary()

print("=== Simulated market, first/last rows ===")
print(summary.head(3))
print(summary.tail(3))

windows = {
    "early (t=0-19)": summary.iloc[0:20],
    "mid   (t=20-39)": summary.iloc[20:40],
    "late  (t=40-60)": summary.iloc[40:],
}

print("\n=== PIN re-estimated from simulated trade tape (Easley-O'Hara MLE) ===")
for label, sub in windows.items():
    res = PINEstimator().fit(sub["buys"].values, sub["sells"].values, n_restarts=6, seed=1)
    true_lambda_avg = sub["lambda_t"].mean()
    print(f"{label:>16s}: true avg lambda_t = {true_lambda_avg:.3f}   "
          f"estimated PIN = {res['PIN']:.3f}")

fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

axes[0].plot(summary.index, summary["lambda_t"], color="darkred", lw=2)
axes[0].set_ylabel("Fraction informed  λ(t)")
axes[0].set_title("Network Information Diffusion (SI model on trader graph)")
axes[0].grid(alpha=0.3)

axes[1].plot(summary.index, summary["price"], color="navy", lw=2, label="Simulated price")
axes[1].axhline(micro.fundamental_value, color="gray", ls="--", label="Fundamental value")
axes[1].set_ylabel("Price")
axes[1].set_title("Price Discovery Driven by Diffusion (Kyle-style price impact)")
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].bar(summary.index, summary["order_flow_imbalance"], color="teal")
axes[2].set_ylabel("Order flow imbalance")
axes[2].set_xlabel("Time step t")
axes[2].set_title("Observable Order Flow Imbalance (buys - sells) / (buys + sells)")
axes[2].grid(alpha=0.3)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "output", "diffusion_model_output.png")
plt.savefig(out_path, dpi=140)
print(f"\nSaved plot to {out_path}")

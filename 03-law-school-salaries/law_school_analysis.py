"""
The R-squared puzzle: why adding a variable can make R-squared fall
-------------------------------------------------------------------

Fits two log-salary regressions on the lawsch85 dataset (156 U.S. law
schools, 1985). The first uses rank and GPA, the second adds age (years
since the school was founded). The R-squared appears to FALL when age is
added, which contradicts the classroom statement that R-squared is
non-decreasing in covariates.

Two things are actually going on:
  1) age has 45 missing values, so adding age shrinks the sample from
     142 to 99 observations. The two models are not fit on the same data.
  2) Even on a common sample, ADJUSTED R-squared can fall. Raw R-squared
     still rises (as theory guarantees), but barely, and the adjustment
     penalty eats the gain because age does not meaningfully predict
     log salary.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# --- 1. Load ---
df = pd.read_csv("data/lawsch85.csv")
print(f"Full sample: {len(df)} law schools")
print("\nMissingness on key variables:")
for c in ["lsalary", "rank", "GPA", "age"]:
    print(f"  {c}: {df[c].isna().sum()} missing")


# --- 2. Model 1: rank and GPA only ---
print("\n" + "=" * 60)
print("MODEL 1: lsalary ~ rank + GPA")
print("=" * 60)
m1 = smf.ols("lsalary ~ rank + GPA", data=df).fit()
print(f"N        = {m1.nobs:.0f}")
print(f"R^2      = {m1.rsquared:.4f}")
print(f"Adj R^2  = {m1.rsquared_adj:.4f}")
print(m1.summary().tables[1])


# --- 3. Model 2: add age ---
print("\n" + "=" * 60)
print("MODEL 2: lsalary ~ rank + GPA + age")
print("=" * 60)
m2 = smf.ols("lsalary ~ rank + GPA + age", data=df).fit()
print(f"N        = {m2.nobs:.0f}")
print(f"R^2      = {m2.rsquared:.4f}")
print(f"Adj R^2  = {m2.rsquared_adj:.4f}")
print(m2.summary().tables[1])


# --- 4. The puzzle, stated ---
print("\n" + "=" * 60)
print("THE PUZZLE")
print("=" * 60)
print(f"Model 1 R^2:  {m1.rsquared:.4f}  (N = {m1.nobs:.0f})")
print(f"Model 2 R^2:  {m2.rsquared:.4f}  (N = {m2.nobs:.0f})")
print("R-squared appears to have fallen when a variable was added.")
print("But notice the samples are different sizes.")


# --- 5. Apples-to-apples: refit Model 1 on Model 2's sample ---
sub = df.dropna(subset=["lsalary", "rank", "GPA", "age"])
m1_same = smf.ols("lsalary ~ rank + GPA", data=sub).fit()

print("\n" + "=" * 60)
print("REFIT ON COMMON SAMPLE (N = {:.0f})".format(m1_same.nobs))
print("=" * 60)
print(f"Model 1 (rank + GPA):          R^2 = {m1_same.rsquared:.4f}   Adj R^2 = {m1_same.rsquared_adj:.4f}")
print(f"Model 2 (rank + GPA + age):    R^2 = {m2.rsquared:.4f}   Adj R^2 = {m2.rsquared_adj:.4f}")
print(f"Change in R^2:       {m2.rsquared - m1_same.rsquared:+.4f}   (must be >= 0, as theory predicts)")
print(f"Change in Adj R^2:   {m2.rsquared_adj - m1_same.rsquared_adj:+.4f}   (can be negative)")


# --- 6. Plots ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: age vs residuals from Model 1 (on the common sample)
ax = axes[0]
sub = sub.assign(resid_m1=sub["lsalary"] - m1_same.predict(sub))
ax.scatter(sub["age"], sub["resid_m1"], alpha=0.65, s=35,
           edgecolor="white", color="#2E5090")
ax.axhline(0, color="#C0392B", lw=1, linestyle="--")
ax.set_xlabel("Age of school (years)")
ax.set_ylabel("Residual from Model 1 (common sample)")
ax.set_title("Does age predict what Model 1 misses?")
ax.grid(alpha=0.3)

# Right: bar chart comparing R-squared and Adj R-squared
ax = axes[1]
labels = ["Model 1\n(rank + GPA)", "Model 2\n(rank + GPA + age)"]
r2 = [m1_same.rsquared, m2.rsquared]
adj_r2 = [m1_same.rsquared_adj, m2.rsquared_adj]
x = np.arange(len(labels))
w = 0.35
b1 = ax.bar(x - w/2, r2, w, label="R²", color="#2E5090")
b2 = ax.bar(x + w/2, adj_r2, w, label="Adjusted R²", color="#C0392B")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.003,
                f"{b.get_height():.4f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("")
ax.set_title("R² vs Adjusted R² on the common sample")
ax.set_ylim(0.78, 0.82)
ax.legend()
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("plots/rsquared_puzzle.png", dpi=140, bbox_inches="tight")
plt.close()
print("\nSaved: plots/rsquared_puzzle.png")

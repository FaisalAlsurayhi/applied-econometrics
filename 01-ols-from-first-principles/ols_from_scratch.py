"""
OLS from first principles
-------------------------

Derive the simple bivariate OLS slope and intercept by hand and verify
against statsmodels. Dataset: Gapminder 2007 cross-section (142 countries).

Model: lifeExp = beta0 + beta1 * log(gdpPercap) + u
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# --- 1. Load and transform ---
df = pd.read_csv("data/gapminder_2007.csv")
df["log_gdp"] = np.log(df["gdpPercap"])
n = len(df)

print(f"Countries in sample: {n}")
print(df[["country", "lifeExp", "gdpPercap", "log_gdp"]].describe().round(2))


# --- 2. Means ---
x_bar = df["log_gdp"].mean()
y_bar = df["lifeExp"].mean()

print(f"\nMean log(GDP per capita): {x_bar:.4f}")
print(f"Mean life expectancy:     {y_bar:.4f}")


# --- 3. Deviations from means ---
df["x_dev"] = df["log_gdp"] - x_bar
df["y_dev"] = df["lifeExp"] - y_bar


# --- 4. Apply the OLS formulas by hand
#
#   beta1_hat = sum_i (x_i - x_bar)(y_i - y_bar) / sum_i (x_i - x_bar)^2
#   beta0_hat = y_bar - beta1_hat * x_bar
#
Sxy = (df["x_dev"] * df["y_dev"]).sum()
Sxx = (df["x_dev"] ** 2).sum()

beta1_hand = Sxy / Sxx
beta0_hand = y_bar - beta1_hand * x_bar

print("\nHand-computed OLS:")
print(f"  sum of cross products  Sxy = {Sxy:.4f}")
print(f"  sum of squared x-devs  Sxx = {Sxx:.4f}")
print(f"  slope      beta1_hat = Sxy / Sxx    = {beta1_hand:.6f}")
print(f"  intercept  beta0_hat = y_bar - beta1_hat*x_bar = {beta0_hand:.6f}")


# --- 5. Verify against statsmodels ---
m = smf.ols("lifeExp ~ log_gdp", data=df).fit()

print("\nstatsmodels OLS:")
print(f"  slope      = {m.params['log_gdp']:.6f}")
print(f"  intercept  = {m.params['Intercept']:.6f}")
print(f"  R-squared  = {m.rsquared:.4f}")

# Max absolute difference between hand result and library result
print(f"\nMax abs difference (hand vs library): "
      f"{max(abs(beta1_hand - m.params['log_gdp']), abs(beta0_hand - m.params['Intercept'])):.2e}")


# --- 6. Residuals ---
df["y_hat"] = beta0_hand + beta1_hand * df["log_gdp"]
df["resid"] = df["lifeExp"] - df["y_hat"]

print(f"\nResidual check (should be near zero):")
print(f"  mean of residuals:  {df['resid'].mean():.2e}")
print(f"  sum of residuals:   {df['resid'].sum():.2e}")

# R-squared from first principles: 1 - SSR / SST
ssr = (df["resid"] ** 2).sum()
sst = (df["y_dev"] ** 2).sum()
r2_hand = 1 - ssr / sst
print(f"  R-squared (by hand): {r2_hand:.4f}")


# --- 7. Plot ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Scatter with regression line
ax = axes[0]
ax.scatter(df["log_gdp"], df["lifeExp"], alpha=0.55, s=35,
           edgecolor="white", color="#2E5090")
xl = np.linspace(df["log_gdp"].min(), df["log_gdp"].max(), 100)
ax.plot(xl, beta0_hand + beta1_hand * xl, color="#C0392B", lw=2,
        label=f"y_hat = {beta0_hand:.1f} + {beta1_hand:.2f} log(gdp)")
ax.set_xlabel("log(GDP per capita)")
ax.set_ylabel("Life expectancy (years)")
ax.set_title("Life expectancy vs log GDP per capita (142 countries, 2007)")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)

# Residuals vs fitted
ax = axes[1]
ax.scatter(df["y_hat"], df["resid"], alpha=0.55, s=35,
           edgecolor="white", color="#2E5090")
ax.axhline(0, color="#C0392B", lw=1, linestyle="--")
ax.set_xlabel("Fitted values")
ax.set_ylabel("Residuals")
ax.set_title("Residuals vs fitted")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("plots/regression_scatter.png", dpi=140, bbox_inches="tight")
plt.close()
print("\nSaved: plots/regression_scatter.png")


# --- 8. Saudi Arabia callout ---
saudi = df[df["country"] == "Saudi Arabia"]
if len(saudi):
    s = saudi.iloc[0]
    pred = beta0_hand + beta1_hand * s["log_gdp"]
    print(f"\nSaudi Arabia (2007):")
    print(f"  GDP per capita     = ${s['gdpPercap']:,.0f}")
    print(f"  Actual life exp.   = {s['lifeExp']:.2f} years")
    print(f"  Predicted life exp. = {pred:.2f} years")
    print(f"  Residual           = {s['lifeExp'] - pred:+.2f} years")

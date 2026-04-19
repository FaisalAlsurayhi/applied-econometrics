"""
Heteroskedasticity in medical spending
--------------------------------------

Estimate a log-linear model of annual medical spending on demographic
and health variables, diagnose heteroskedasticity in the residuals, and
compare inference under default (OLS) vs heteroskedasticity-robust
standard errors.

Data: 2003 Medical Expenditure Panel Survey (MEPS), U.S. adults aged
65 and older, restricted to those with positive annual expenditure.
Sample size: 2,955.

Model:
    ln(totexp) = b0 + b1 age + b2 totchr + b3 income + b4 female
                 + b5 suppins + b6 phylim + b7 actlim + u
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan, het_white


# --- 1. Load ---
df = pd.read_csv("data/mus03data.csv")
print(f"Sample size: {len(df)}")
print(df[["lntotexp", "age", "totchr", "income", "female",
          "suppins", "phylim", "actlim"]].describe().round(3))


# --- 2. OLS with default standard errors ---
formula = "lntotexp ~ age + totchr + income + female + suppins + phylim + actlim"
m_ols = smf.ols(formula, data=df).fit()

print("\n" + "=" * 70)
print("OLS with default (non-robust) standard errors")
print("=" * 70)
print(m_ols.summary().tables[1])
print(f"R^2 = {m_ols.rsquared:.4f}")


# --- 3. Residuals-vs-fitted plot ---
fitted = m_ols.fittedvalues
resid = m_ols.resid

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.scatter(fitted, resid, alpha=0.35, s=15,
           edgecolor="none", color="#2E5090")
ax.axhline(0, color="#C0392B", lw=1, linestyle="--")
ax.set_xlabel("Fitted values")
ax.set_ylabel("Residuals")
ax.set_title("Residuals vs fitted")
ax.grid(alpha=0.3)

# --- 4. Squared residuals vs fitted (makes heteroskedasticity more visible) ---
ax = axes[1]
ax.scatter(fitted, resid**2, alpha=0.35, s=15,
           edgecolor="none", color="#2E5090")
# Overlay a smoothed trend line
from numpy.polynomial import polynomial as P
order = np.argsort(fitted)
# Bin-means for a quick trend
bins = pd.cut(fitted, bins=20)
trend = pd.DataFrame({"fitted": fitted, "r2": resid**2, "bin": bins})
binned = trend.groupby("bin", observed=True).agg({"fitted": "mean", "r2": "mean"})
ax.plot(binned["fitted"], binned["r2"], color="#C0392B", lw=2,
        label="binned mean")
ax.set_xlabel("Fitted values")
ax.set_ylabel("Squared residuals")
ax.set_title("Squared residuals vs fitted")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("plots/heteroskedasticity_diagnostics.png",
            dpi=140, bbox_inches="tight")
plt.close()
print("\nSaved: plots/heteroskedasticity_diagnostics.png")


# --- 5. Formal tests ---
print("\n" + "=" * 70)
print("Formal heteroskedasticity tests")
print("=" * 70)

# Breusch-Pagan
bp_stat, bp_p, _, _ = het_breuschpagan(resid, m_ols.model.exog)
print(f"\nBreusch-Pagan test:")
print(f"  LM statistic = {bp_stat:.3f}")
print(f"  p-value      = {bp_p:.4g}")
print(f"  Interpretation: " +
      ("reject homoskedasticity" if bp_p < 0.05 else "fail to reject"))

# White (more general — includes interactions and squares)
white_stat, white_p, _, _ = het_white(resid, m_ols.model.exog)
print(f"\nWhite test:")
print(f"  LM statistic = {white_stat:.3f}")
print(f"  p-value      = {white_p:.4g}")
print(f"  Interpretation: " +
      ("reject homoskedasticity" if white_p < 0.05 else "fail to reject"))


# --- 6. Refit with robust SEs (HC1, Stata's default) ---
m_rob = smf.ols(formula, data=df).fit(cov_type="HC1")

print("\n" + "=" * 70)
print("OLS with heteroskedasticity-robust standard errors (HC1)")
print("=" * 70)
print(m_rob.summary().tables[1])


# --- 7. Side-by-side comparison of SEs ---
comp = pd.DataFrame({
    "coef": m_ols.params.round(4),
    "se_default": m_ols.bse.round(4),
    "se_robust": m_rob.bse.round(4),
    "t_default": m_ols.tvalues.round(3),
    "t_robust": m_rob.tvalues.round(3),
    "p_default": m_ols.pvalues.round(4),
    "p_robust": m_rob.pvalues.round(4),
})
comp["se_pct_change"] = (100 * (m_rob.bse - m_ols.bse) / m_ols.bse).round(2)

print("\n" + "=" * 70)
print("Side-by-side: default vs robust")
print("=" * 70)
print(comp)


# --- 8. Economic interpretation ---
print("\n" + "=" * 70)
print("Interpretation of key coefficients (robust SEs)")
print("=" * 70)
for var, desc in [("totchr", "each additional chronic condition"),
                  ("suppins", "having supplemental insurance")]:
    coef = m_rob.params[var]
    # For log-dependent variable, percent change = 100*(exp(coef) - 1)
    pct = 100 * (np.exp(coef) - 1)
    print(f"  {var} ({desc}): coef={coef:+.4f}  =>  "
          f"{pct:+.1f}% difference in expected expenditure")

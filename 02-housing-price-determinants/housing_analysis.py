"""
Housing price determinants
--------------------------

Multiple regression of house price on bedrooms and square footage, using
the hprice1 dataset from Wooldridge (1990, Boston suburbs, 88 houses).

The analytical point is that bivariate and multiple regression tell
different stories about the bedroom coefficient. Once square footage is
controlled for, the value of an extra bedroom changes substantially,
because bedrooms and total size are correlated.

Columns in hprice1:
    price   house price, $1000s
    bdrms   number of bedrooms
    sqrft   square footage
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# --- 1. Load ---
df = pd.read_csv("data/hprice1.csv")
print(f"Sample: {len(df)} houses")
print(df[["price", "bdrms", "sqrft"]].describe().round(2))


# --- 2. Bivariate regression: price ~ bdrms ---
print("\n" + "=" * 60)
print("BIVARIATE: price ~ bdrms")
print("=" * 60)
m_biv = smf.ols("price ~ bdrms", data=df).fit()
print(m_biv.summary().tables[1])
print(f"R-squared: {m_biv.rsquared:.4f}")


# --- 3. Multiple regression: price ~ bdrms + sqrft ---
print("\n" + "=" * 60)
print("MULTIPLE: price ~ bdrms + sqrft")
print("=" * 60)
m_mult = smf.ols("price ~ bdrms + sqrft", data=df).fit()
print(m_mult.summary().tables[1])
print(f"R-squared:      {m_mult.rsquared:.4f}")
print(f"F-statistic:    {m_mult.fvalue:.2f}  (p = {m_mult.f_pvalue:.2e})")


# --- 4. Compare bedroom coefficients across the two models ---
print("\n" + "=" * 60)
print("COEFFICIENT COMPARISON")
print("=" * 60)
print(f"Bedroom coefficient (bivariate):  ${m_biv.params['bdrms']*1000:>10,.0f}")
print(f"Bedroom coefficient (with sqrft): ${m_mult.params['bdrms']*1000:>10,.0f}")
print(f"Correlation between bdrms and sqrft: {df[['bdrms','sqrft']].corr().iloc[0,1]:.3f}")


# --- 5. The friend's house question ---
# Friend bought: 4 bedrooms, 2,438 sqft, paid $300,000
print("\n" + "=" * 60)
print("PREDICTION: friend's house (4 bdrms, 2438 sqft)")
print("=" * 60)
new_house = pd.DataFrame({"bdrms": [4], "sqrft": [2438]})
pred = m_mult.get_prediction(new_house).summary_frame(alpha=0.05)
print(pred.round(2))

predicted_price = pred["mean"].iloc[0] * 1000
paid = 300_000
lower = pred["obs_ci_lower"].iloc[0] * 1000
upper = pred["obs_ci_upper"].iloc[0] * 1000

print(f"\nModel prediction:  ${predicted_price:>10,.0f}")
print(f"Amount paid:       ${paid:>10,.0f}")
print(f"Difference:        ${paid - predicted_price:>+10,.0f}")
print(f"95% prediction interval for a house with these characteristics:")
print(f"    ${lower:,.0f} to ${upper:,.0f}")
print(f"$300,000 falls {'inside' if lower <= paid <= upper else 'outside'} the prediction interval.")


# --- 6. Plots ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: price vs sqrft, colored by bedrooms
ax = axes[0]
scatter = ax.scatter(df["sqrft"], df["price"], c=df["bdrms"],
                     cmap="viridis", alpha=0.75, s=45, edgecolor="white")
ax.set_xlabel("Square footage")
ax.set_ylabel("Price ($1,000s)")
ax.set_title("Price vs square footage (color = bedrooms)")
cbar = plt.colorbar(scatter, ax=ax, label="Bedrooms")
ax.grid(alpha=0.3)

# Right: predicted vs actual
ax = axes[1]
df["pred"] = m_mult.predict(df)
ax.scatter(df["pred"], df["price"], alpha=0.75, s=45,
           edgecolor="white", color="#2E5090")
lims = [min(df["pred"].min(), df["price"].min()),
        max(df["pred"].max(), df["price"].max())]
ax.plot(lims, lims, color="#C0392B", lw=1.5, linestyle="--",
        label="perfect prediction")
ax.set_xlabel("Predicted price ($1,000s)")
ax.set_ylabel("Actual price ($1,000s)")
ax.set_title("Predicted vs actual prices")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("plots/housing_regression.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"\nSaved: plots/housing_regression.png")

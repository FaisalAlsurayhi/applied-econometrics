# Applied Econometrics

This repo is a set of applied econometrics projects, each taking a real dataset and working through a different method end-to-end. The pieces are organized from foundation (how OLS actually computes its estimates) to inference (what happens when its assumptions fail).

Every project has a research question, a worked analysis in Python, and a writeup in its own README. I tried to write the way I would explain a finding to someone, not the way a textbook would.

## What This Shows

This repo is meant to show the statistics/econometrics side of my analytics work. The focus is not just running a model, but explaining what the coefficient means, whether the result is reliable, and what can go wrong if the assumptions are ignored.

For data analyst and reporting analyst roles, the main skills demonstrated are:

- Regression analysis and interpretation
- Model comparison and adjusted R-squared reasoning
- Missing-data awareness when comparing specifications
- Heteroskedasticity testing with Breusch-Pagan and White tests
- Robust standard errors and practical inference
- Clear written communication of technical results

## Projects

**1. [OLS from first principles](01-ols-from-first-principles/)**
Bivariate OLS derived by hand from Σ(xᵢ-x̄)(yᵢ-ȳ)/Σ(xᵢ-x̄)², using Gapminder's 2007 cross-section of 142 countries to estimate the relationship between income and life expectancy. My hand-calculated slope matches `statsmodels` to 16 decimal places. Doubling GDP per capita is associated with about 5 additional years of life expectancy on average across countries.

**2. [Housing price determinants](02-housing-price-determinants/)**
Multiple regression on 88 Boston suburban houses. The bedroom coefficient falls from $62,000 in a bivariate specification to $15,000 once square footage is added, and loses statistical significance entirely. This is the clean illustration of why multiple regression matters: bivariate bedroom coefficients mostly reflect the correlation between bedrooms and total size, not the pure value of an extra bedroom.

**3. [The R-squared puzzle: law school salaries](03-law-school-salaries/)**
Adding a covariate to a log-salary regression makes R² appear to fall, which contradicts the classroom result that R² is non-decreasing in regressors. Two things are actually happening: the sample changes because the new variable has missing values, and adjusted R² can fall even on a fixed sample if the added variable does not pull its weight against the degrees-of-freedom penalty.

**4. [Heteroskedasticity in medical spending](04-heteroskedasticity-medical-spending/)**
OLS with formal diagnostics on 2,955 elderly respondents from the 2003 Medical Expenditure Panel Survey. Breusch-Pagan and White tests strongly reject homoskedasticity, but robust standard errors move by only 1 to 3 percent and no coefficient flips significance. The honest takeaway is not that the original OLS inference was invalid, but that robust SEs are the defensible default when you have tested and found the assumption violated.

## Methods arc

The four projects map to a progression from estimation mechanics to inference under assumption failure:

- Project 1 shows how OLS actually computes β from the data
- Project 2 shows what multiple regression adds that bivariate regression cannot
- Project 3 shows how to compare nested models honestly, and what R² actually tells us
- Project 4 shows how to diagnose a broken assumption and what to do about it

## Tools

`Python` · `pandas` · `statsmodels` · `numpy` · `plotly` · `econometrics`

## Data and attribution

The datasets are publicly available and widely used in introductory econometrics:

- **Gapminder 2007** cross-section, accessed via `plotly.express.data.gapminder()`
- **Wooldridge** `hprice1` and `lawsch85` from *Introductory Econometrics: A Modern Approach*
- **MEPS 2003** extract `mus03data` from Cameron and Trivedi, *Microeconometrics Using Stata*

Each project's README has a specific source citation. The research questions, analyses, and writeups are my own.

## About

I'm Faisal Alsurayhi, a BA Economics student at George Mason University (class of 2026), building a data analyst portfolio ahead of my return to Saudi Arabia's Eastern Province this summer. Related portfolio work:

- [crude-oil-price-analysis](https://github.com/FaisalAlsurayhi/crude-oil-price-analysis) — time-series analysis of crude oil prices in Python
- [nitaqat-workforce-sql](https://github.com/FaisalAlsurayhi/nitaqat-workforce-sql) — SQL analysis of Saudi labor nationalization policy
- [data-analytics-portfolio](https://github.com/FaisalAlsurayhi/data-analytics-portfolio) — mixed SQL, Excel, and Power BI projects

Find me on [LinkedIn](https://www.linkedin.com/in/faisal-alsurayhi-64049422b/).

# methane-eos-regression

A data-driven equation of state for methane (CH₄), learned by regression from four experimental P–ρ–T datasets. The model predicts the compressibility factor **Z = P / (ρRT)** from reduced density and temperature, is validated on data it has never seen, and is published as an interactive web page.

## Overview

Instead of using a fixed analytical EOS (van der Waals, Peng–Robinson, …), the project builds a library of polynomial terms in density and temperature and lets regression choose and weight them.

- **Target:** Z − 1
- **Inputs:** δ = ρ / ρc (reduced density) and τ = Tc / T (inverse reduced temperature)
- **Constants:** Tc = 190.564 K, ρc = 162.66 kg/m³, M = 0.01604246 kg/mol, R = 8.314462618 J/(mol·K)
- **Term bank:** δᵏ τᵗ for k = 1…6 and t = 0…5 (36 terms)

Z − 1 = Σ cᵢ · δᵏ τᵗ

The **final compact model** is a lasso fit that keeps **15 of the 36 terms**.

## Data

Raw files in `data/raw/` are never edited. Unit conversion, flagging and cleaning are done in the notebook.

| File | Source | Region | Usable points |
|---|---|---|---|
| `douslin_1964.csv` | Douslin et al., 1964 | gas / supercritical | 171 |
| `vennix_1965.csv` | Vennix et al., 1965 | liquid / dense fluid | 278 |
| `goodwin_prydz_1972.csv` | Goodwin & Prydz, 1972 | cold liquid | 300 |
| `cristancho_2010.csv` | Cristancho et al., 2010 | gas / supercritical, high pressure | 83 |

- **835 points in total, 832 used** for fitting
- **Range:** T ≈ 90.7–450.1 K, P ≈ 11.7 kPa–188 MPa
- **3 rows excluded** (kept in the raw files, flagged with `exclude_reason`): the critical-point row, the triple-point row (both Goodwin & Prydz), and one suspect outlier in a Vennix isochore

## Models compared

| Model | Description |
|---|---|
| Ideal gas | Z = 1 (baseline) |
| Virial OLS | 5-term virial baseline: δ, δτ, δτ², δ², δ²τ |
| Ridge | all 36 terms, L2 penalty (RidgeCV) |
| Lasso | L1 penalty, selects a sparse set of terms |
| Bayesian ridge | ridge with predictive uncertainty |
| ARD | Bayesian model with per-term relevance |

Features are standardised using the training data only. Regularisation strength is tuned with group cross-validation *inside* the training sources, never on the test source.

## Validation

All errors are on data **not used for training**.

1. **Leave-one-source-out:** train on three datasets, test on the fourth.
2. **Leave-one-isotherm/isochore-out:** 50 held-out groups (whole isotherms for Cristancho and Douslin, whole isochores for Goodwin & Prydz and Vennix).
3. **High-pressure test:** train on P < 40 MPa, test on P ≥ 40 MPa (63 rows).
4. **Near-critical test:** within 10 K of Tc and 0.6 ≤ δ ≤ 1.6 (67 rows, all from Vennix). The model is trained *without* these rows. The rule was fixed before looking at the results.

## Results

Median % error in Z (lower is better).

**Leave-one-source-out**

| Model | Cristancho | Douslin | Goodwin & Prydz | Vennix |
|---|---|---|---|---|
| Ideal gas | 18.50 | 8.85 | 56.53 | 81.46 |
| Virial OLS | 7.21 | 3.96 | 170.61 | 48.71 |
| Ridge | 0.50 | 0.38 | 62.91 | 1.89 |
| Lasso | 0.87 | 1.25 | 9.75 | 2.03 |
| Bayesian ridge | 0.30 | 0.13 | 11.56 | 2.00 |
| ARD | 0.36 | 0.18 | 119.58 | 1.94 |

**Held-out isotherms / isochores**

| Model | Cristancho | Douslin | Goodwin & Prydz | Vennix |
|---|---|---|---|---|
| Ideal gas | 18.50 | 8.85 | 56.53 | 81.46 |
| Virial OLS | 7.19 | 3.91 | 19.15 | 15.58 |
| Ridge | 0.15 | 0.07 | 0.51 | 0.46 |
| Lasso (15 terms) | 0.53 | 0.55 | 0.67 | 1.41 |

**High pressure (P ≥ 40 MPa, trained below 40 MPa)**

| Model | Cristancho | Douslin | Vennix |
|---|---|---|---|
| Ideal gas | 38.24 | 8.28 | 22.70 |
| Virial OLS | 17.78 | 6.61 | 19.00 |
| Ridge | 0.80 | 0.03 | 0.74 |
| Lasso (15 terms) | 2.33 | 0.76 | 0.36 |

**Near the critical point (trained without these rows)**

| Model | Median % error | Worst error |
|---|---|---|
| Ideal gas | 256.79 | 439.9 |
| Virial OLS | 221.55 | 271.5 |
| Ridge | 9.52 | 18.0 |
| Lasso (15 terms) | 13.32 | 29.3 |

### Main findings

- Both regression models are far better than the ideal gas and the 5-term virial baseline everywhere tested.
- Ridge is slightly more accurate; the 15-term lasso is nearly as good while being compact and readable, so it is used as the final model.
- Predicting an entire unseen phase region is hard: leaving out Goodwin & Prydz (cold liquid) gives large errors because no other dataset covers those conditions. Holding out single isotherms/isochores instead gives errors around 0.5–1.4 %.
- Near the critical point, errors rise to roughly 10–13 % (median).
- Bayesian ridge uncertainties are over-confident: 95 % intervals covered only 71 %, 100 %, 68 % and 23 % of held-out points (Cristancho, Douslin, Goodwin & Prydz, Vennix).

### Compact lasso model (α = 2×10⁻⁵, 15 terms)

`d1_t1, d2_t0, d6_t1, d6_t0, d1_t0, d3_t2, d6_t2, d1_t2, d1_t3, d1_t4, d6_t3, d6_t4, d4_t1, d3_t1, d6_t5`

where `dK_tT` means δᵏ τᵗ. Coefficients are applied to standardised features and are stored in `docs/model.json`.

## Repository structure

```
methane-eos-regression/
├── data/raw/                 # four experimental datasets (unmodified)
├── docs/
│   ├── index.html            # interactive web page
│   └── model.json            # fitted models, lasso path, validation tables, data points
├── notebooks/
│   └── Untitled26.ipynb      # data prep, regression, validation
├── export_for_website.py     # writes docs/model.json
└── README.md
```

## Getting started

The notebook is written for Google Colab and clones this repository automatically. To run locally:

```bash
git clone https://github.com/somya-assery/methane-eos-regression.git
cd methane-eos-regression
pip install numpy pandas scikit-learn matplotlib jupyter
jupyter notebook notebooks/
```

Run the notebook from the top, then run `export_for_website.py` in the same session (it uses the `usable` table and the constants `TC` and `RHOC` defined in the data-preparation step). It writes `docs/model.json`, which `docs/index.html` reads.

## Website

The `docs/` folder is served with GitHub Pages (Settings → Pages → source: `main` / `docs`). The page shows the experimental points, the fitted models, the lasso sparsity path and the validation results.

## Limitations

- Valid only for T ≈ 90–450 K and P up to ≈ 188 MPa; no extrapolation outside the data
- Accuracy drops near the critical point
- Entire phase regions that no other dataset covers (e.g. cold liquid) cannot be predicted reliably
- The lasso strength was picked from the leave-one-source-out path, so the leave-one-source-out numbers for the final lasso are slightly optimistic
- No thermodynamic-consistency constraints are imposed on the fit

## References

1. Douslin, D. R., et al. (1964). P–V–T data for methane.
2. Vennix, A. J., et al. (1965). Volumetric properties of methane.
3. Goodwin, R. D., & Prydz, R. (1972). P–V–T data for methane.
4. Cristancho, D. E., et al. (2010). Density measurements of methane.

*(Add full citations and DOIs.)*

## Authors

Somya Assery (25110303)
Patel Preet  (25110235)

## License

MIT

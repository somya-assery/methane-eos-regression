# methane-eos-regression

**ES 335 final project.** A data-driven equation of state for methane (CH₄), learned by regression from four experimental P–ρ–T datasets and tested on data the models never saw.

**Live site:** https://somya-assery.github.io/methane-eos-regression/
**Source code:** https://github.com/somya-assery/methane-eos-regression

## Question

Given the temperature and molar density of pure methane, how accurately can sparse and Bayesian regression predict the measured compressibility factor **Z = P / (ρRT)**, when tested on held-out labs and regions? Every model is judged only against the raw measurements. One row is one measured state point (T, P, ρ).

## Overview

Instead of using a fixed analytical equation of state, the project builds a library of polynomial terms in density and temperature and lets regression choose and weight them.

- **Target:** Z − 1
- **Inputs:** δ = ρ / ρc (reduced density) and τ = Tc / T (inverse reduced temperature)
- **Constants:** Tc = 190.564 K, ρc = 162.66 kg/m³, M = 0.01604246 kg/mol, R = 8.314462618 J/(mol·K) (critical constants from Setzmann & Wagner, 1991)
- **Term bank:** δᵏ τᵗ for k = 1…6 and t = 0…5 (36 terms), so Z − 1 = Σ cᵢ · δᵏ τᵗ

The final compact model is a lasso fit that keeps **15 of the 36 terms**.

## Data

Raw files are in `data/raw/`. They are not modified after transcription. Unit conversion, flagging and cleaning are done in the notebook.

| File | Source | Region | Usable points |
|---|---|---|---|
| `cristancho_2010.csv` | Cristancho et al., 2010 | gas / supercritical, high pressure | 83 |
| `douslin_1964.csv` | Douslin et al., 1964 | gas / supercritical | 171 |
| `goodwin_prydz_1972.csv` | Goodwin & Prydz, 1972 | cold liquid | 300 |
| `vennix_1965.csv` | Vennix, 1965 (PhD thesis) | liquid / dense fluid | 278 |

- **835 rows in the raw files, 832 used.** Usable range: T ≈ 92–450 K, P ≈ 0.35–188 MPa.
- **3 rows excluded** (kept in the raw files and flagged with `exclude_reason`): the critical-point row and the triple-point row (both in Goodwin & Prydz), and one suspect Vennix point (18.411 mol/L, 272.927 K, 498.737 bar) whose pressure jump is out of line with the rest of its isochore. Its printed deviation columns are consistent with the printed pressure, so it may be a genuine anomaly in the source table.

### Data provenance and access

- **Cristancho 2010** was converted from the NIST ThermoML JSON file (https://trc.nist.gov/ThermoML/10.1021/je9004849.html). Pressure was converted from kPa to MPa, and a column with the stated density uncertainty was added.
- **Douslin 1964, Vennix 1965 and Goodwin & Prydz 1972** were copied from Tables 4, 3 and 1 of Goodwin & Prydz (1972), which is open access on PubMed Central (PMC6706564). Douslin and Vennix are therefore secondary sources, and the Vennix thesis is not peer reviewed.
- The authors' fitted-equation columns (P CALC and the percent deviations) were **not copied**, so they cannot leak into the features.
- One stray character in Goodwin & Prydz row 1308 was fixed. Row counts and isochore-smoothness checks were run to catch transcription errors.
- Labs differ in units, temperature scales (1960s scales versus ITS-90) and stated uncertainty, and rows on one isotherm or isochore are nearly copies of each other, so the effective sample size is much smaller than 832.

## Models compared

| Model | Description |
|---|---|
| Ideal gas | Z = 1 (baseline) |
| Virial OLS | 5-term virial baseline: δ, δτ, δτ², δ², δ²τ |
| Ridge | all 36 terms, L2 penalty (`RidgeCV`) |
| Lasso | L1 penalty, selects a sparse set of terms |
| Bayesian ridge | ridge with predictive uncertainty |
| ARD | Bayesian model with per-term relevance |

Features are standardised using training data only. In the leave-one-source-out test, regularisation strength is tuned with group cross-validation inside the training sources, never on the test source. In the isotherm, high-pressure and near-critical tests, ridge uses `RidgeCV` with its built-in cross-validation on the training rows only (not grouped by source), and lasso uses the fixed strength α = 2×10⁻⁵. Bayesian ridge and ARD were evaluated in the leave-one-source-out test only.

## Validation

All errors are on data **not used for training**.

1. **Leave-one-source-out:** train on three datasets, test on the fourth.
2. **Leave-one-isotherm/isochore-out:** 50 held-out groups (whole isotherms for Cristancho and Douslin, whole isochores for Goodwin & Prydz and Vennix).
3. **High-pressure test:** train on P < 40 MPa, test on P ≥ 40 MPa (63 rows).
4. **Near-critical test:** rows within 10 K of Tc and with 0.6 ≤ δ ≤ 1.6 (67 rows, all from Vennix). The models are trained without these rows, and the rule was fixed before the results were seen.

The metric is the median percent error in Z, because an average hides regions where a model fails. Pressure error is not used in dense liquid, where small density errors become large pressure errors.

## Results

Median % error in Z (lower is better).

**Leave-one-source-out**

| Model | Cristancho | Douslin | Goodwin & Prydz | Vennix |
|---|---|---|---|---|
| Ideal gas | 18.50 | 8.85 | 56.53 | 81.46 |
| Virial OLS | 7.21 | 3.96 | 170.61 | 48.71 |
| Ridge | 0.50 | 0.38 | 62.91 | 1.89 |
| Lasso (CV-tuned, 36 terms) | 0.87 | 1.25 | 9.75 | 2.03 |
| Lasso (15 terms) | 1.40 | 1.13 | 5.05 | 2.18 |
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

Douslin has very few rows above 40 MPa, so its high-pressure column rests on a small sample.

**Near the critical point (trained without these rows; Vennix only)**

| Model | Median % error | Worst error |
|---|---|---|
| Ideal gas | 256.79 | 439.9 |
| Virial OLS | 221.55 | 271.5 |
| Ridge | 9.52 | 18.0 |
| Lasso (15 terms) | 13.32 | 29.3 |

### Main findings

- Both regression models beat the ideal gas and the virial baseline in every test except the leave-one-source-out fold that hides the cold-liquid data (Goodwin & Prydz), where ridge and ARD extrapolate worse than the ideal gas.
- Ridge is slightly more accurate, and the 15-term lasso is nearly as good while being compact and readable, so it is used as the final model. The price of compactness is roughly 0.4–1% in median error.
- Predicting an entire unseen phase region is hard: leaving out Goodwin & Prydz (cold liquid) gives large errors because no other dataset covers those conditions. Holding out single isotherms or isochores instead gives median errors of 0.07–1.4%.
- Near the critical point, median errors rise to roughly 10–13%. Only one lab covers this region, so the result is not cross-checked.
- Bayesian ridge uncertainties are over-confident on three of the four held-out labs: its 95% intervals covered 71% (Cristancho), 68% (Goodwin & Prydz) and 23% (Vennix) of held-out points, and 100% for Douslin, where they were too wide.

### Compact lasso model (α = 2×10⁻⁵, 15 terms)

`d1_t1, d2_t0, d6_t1, d6_t0, d1_t0, d3_t2, d6_t2, d1_t2, d1_t3, d1_t4, d6_t3, d6_t4, d4_t1, d3_t1, d6_t5`

where `dK_tT` means δᵏ τᵗ. Coefficients apply to standardised features and are stored in `docs/model.json`.

## Repository structure

```
methane-eos-regression/
├── data/raw/                 # four experimental datasets (as transcribed)
├── docs/
│   ├── index.html            # interactive web page
│   └── model.json            # fitted models, lasso path, results, data points
├── notebooks/
│   └── methane_eos_regression.ipynb   # data prep, regression, validation
├── export_for_website.py     # writes docs/model.json
└── README.md
```

## How to reproduce

1. Open `notebooks/methane_eos_regression.ipynb` in Google Colab (File → Open notebook → GitHub). The first cell clones this repository.
2. Choose **Runtime → Restart session and run all**. The notebook builds the table of usable rows (`usable`), fits all models and prints the tables above. Random seeds are not needed, since all fits are deterministic.
3. To regenerate the website data, run `%run -i export_for_website.py` in the same session. The `-i` flag makes the script use the variables already defined in the notebook (`usable`, `TC`, `RHOC`). It writes `docs/model.json`.
4. In Colab, files disappear when the session ends, so download `docs/model.json` and upload it to the repository.

To run locally instead:

```bash
git clone https://github.com/somya-assery/methane-eos-regression.git
cd methane-eos-regression
pip install numpy pandas scikit-learn matplotlib jupyter
jupyter notebook notebooks/
```

Lasso results can differ in the last digit between scikit-learn versions.

## Website

The `docs/` folder is served with GitHub Pages (Settings → Pages → `main` / `docs`). The page lets a visitor pick a model, change the number of lasso terms and the temperature, and compare the fitted curve with the real measurements. It shows the leave-one-source-out and near-critical results. The isotherm, high-pressure and uncertainty-coverage results are in this README and the notebook.

## Limitations

- Valid only for T ≈ 92–450 K and P up to ≈ 188 MPa for pure methane; no extrapolation outside the data.
- Only four sources, two from the 1960s and one an unreviewed thesis, and three of them are secondary transcriptions.
- Accuracy drops near the critical point, and only one lab covers that region.
- Entire phase regions that no other dataset covers (for example cold liquid) cannot be predicted reliably.
- The lasso strength (15 terms) was chosen from the leave-one-source-out path, so the leave-one-source-out numbers for the final lasso are slightly optimistic.
- Bayesian uncertainty bands are over-confident on unseen labs, so they are not shown as error bars.
- No thermodynamic-consistency constraints are imposed, and the fitted terms are an empirical summary, not a physical law.
- Different labs use different temperature scales and have different uncertainties, which the models do not account for.

## References

1. Cristancho, D. E., Mantilla, I. D., Ejaz, S., & Hall, K. R. (2010). Accurate PVT data for methane from (300 to 450) K up to 180 MPa. *J. Chem. Eng. Data*, 55(2), 826–829. https://doi.org/10.1021/je9004849. Data from NIST ThermoML.
2. Douslin, D. R., Harrison, R. H., Moore, R. T., & McCullough, J. P. (1964). P-V-T relations for methane. *J. Chem. Eng. Data*, 9(3), 358. Data as tabulated in reference 3.
3. Goodwin, R. D., & Prydz, R. (1972). Densities of compressed liquid methane, and the equation of state. *J. Res. Natl. Bur. Stand. A*, 76(1), 81. PMCID: PMC6706564.
4. Vennix, A. J. (1965). *Low temperature volumetric properties and development of an equation of state for methane.* PhD thesis, Rice University, Houston, TX. Data as tabulated in reference 3.
5. Setzmann, U., & Wagner, W. (1991). A new equation of state and tables of thermodynamic properties for methane covering the range from the melting line to 625 K at pressures up to 1000 MPa. *J. Phys. Chem. Ref. Data*, 20(6), 1061. Used for the critical constants.

## AI-use disclosure

I used Claude (Anthropic) to help plan the project, find and assess data sources, write and debug the Python code (data loading, term bank, evaluation, website export), and draft the website . I collected and checked the data, ran all code in Google Colab, compared the outputs with my own runs, and I am responsible for the content.

## Authors

Somya Assery (25110303)
Patel Preet (25110235)

## License

MIT

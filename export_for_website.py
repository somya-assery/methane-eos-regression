# Writes docs/model.json for the website (docs/index.html).
# Run from the repository root, either as a Colab cell after the data-preparation step,
# or on its own:  python export_for_website.py
# If `usable`, `TC` and `RHOC` already exist (notebook session) they are reused;
# otherwise they are rebuilt from data/raw with exactly the same preprocessing.
import json, os, warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, RidgeCV, BayesianRidge, ARDRegression, LinearRegression
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")

if not os.path.isdir("data/raw") and os.path.isdir("methane-eos-regression"):
    os.chdir("methane-eos-regression")

def build_usable():
    R, M = 8.314462618, 0.01604246            # J/(mol K), kg/mol
    tc, rhoc, raw = 190.564, 162.66 / 0.01604246, "data/raw"
    cri, dou = pd.read_csv(f"{raw}/cristancho_2010.csv"), pd.read_csv(f"{raw}/douslin_1964.csv")
    goo, ven = pd.read_csv(f"{raw}/goodwin_prydz_1972.csv"), pd.read_csv(f"{raw}/vennix_1965.csv")
    cri_si = pd.DataFrame({"T_K": cri.T_K, "P_Pa": cri.P_MPa*1e6, "rho_mol_m3": cri.rho_kg_m3/M})
    dou_si = pd.DataFrame({"T_K": dou.T_K, "P_Pa": dou.P_bar*1e5, "rho_mol_m3": dou.rho_mol_L*1e3})
    goo_si = pd.DataFrame({"T_K": goo.T_K, "P_Pa": goo.P_bar*1e5, "rho_mol_m3": goo.rho_mol_L*1e3})
    ven_si = pd.DataFrame({"T_K": ven.T_K, "P_Pa": ven.P_bar*1e5, "rho_mol_m3": ven.rho_mol_L*1e3})
    cri_si["source"], cri_si["region"] = "Cristancho 2010", "gas/supercritical"
    dou_si["source"], dou_si["region"] = "Douslin 1964", "gas/supercritical"
    goo_si["source"], goo_si["region"] = "Goodwin & Prydz 1972", "cold liquid"
    ven_si["source"], ven_si["region"] = "Vennix 1965", "liquid/dense fluid"
    goo_si["exclude_reason"] = ""
    goo_si.loc[(goo.id == 1) & (goo.T_K > 150), "exclude_reason"] = "critical point row"
    goo_si.loc[(goo.id == 1) & (goo.T_K < 150), "exclude_reason"] = "triple point row"
    ven_si["exclude_reason"] = ""
    ven_si.loc[(ven.rho_mol_L == 18.411) & (ven.T_K == 272.927), "exclude_reason"] = "suspect: outlier in isochore"
    df = pd.concat([cri_si, dou_si, goo_si, ven_si], ignore_index=True)
    df["exclude_reason"] = df["exclude_reason"].fillna("")
    df["Z"] = df.P_Pa / (df.rho_mol_m3 * R * df.T_K)
    df["delta"] = df.rho_mol_m3 / rhoc
    df["tau"] = tc / df.T_K
    return df[df.exclude_reason == ""].copy(), tc, rhoc

if not all(k in globals() for k in ("usable", "TC", "RHOC")):
    usable, TC, RHOC = build_usable()

# ---------------------------------------------------------------- data and features
D = usable.reset_index(drop=True); y = (D.Z - 1).values; src = D.source.values
SRC = sorted(D.source.unique())
assert len(D) == 832, f"expected 832 usable rows, got {len(D)}"
names = [f"d{k}_t{t}" for k in range(1, 7) for t in range(6)]
X = np.column_stack([D.delta.values**k * D.tau.values**t for k in range(1, 7) for t in range(6)])
Xv = np.column_stack([D.delta.values, D.delta.values*D.tau.values, D.delta.values*D.tau.values**2,
                      D.delta.values**2, D.delta.values**2*D.tau.values])
# near-critical rule (fixed in advance): within 10 K of Tc and 0.6 <= delta <= 1.6
near = (((D.T_K - TC).abs() < 10) & D.delta.between(0.6, 1.6)).values
ALPHA = 2.0e-5                                   # final compact lasso (15 terms)

# ---------------------------------------------------------------- models
def mk(name, alpha=None):
    if name == "ridge": return RidgeCV(alphas=np.logspace(-6, 3, 30))
    if name == "lasso": return Lasso(alpha=alpha, max_iter=200000)
    if name == "bayesian ridge": return BayesianRidge()
    if name == "ARD": return ARDRegression(max_iter=300)

def pred(name, tr, te, alpha=None, grouped=False):
    if name == "ideal gas": return np.zeros(te.sum())
    if name == "virial OLS": return LinearRegression(fit_intercept=False).fit(Xv[tr], y[tr]).predict(Xv[te])
    s = StandardScaler().fit(X[tr])              # scaler fitted on training rows only
    A, B = s.transform(X[tr]), s.transform(X[te])
    if name == "ridge" and grouped:              # tuning inside the training sources (as in the notebook)
        inner = list(GroupKFold(n_splits=len(np.unique(src[tr]))).split(A, y[tr], src[tr]))
        return RidgeCV(alphas=np.logspace(-6, 3, 30), cv=inner).fit(A, y[tr]).predict(B)
    return mk(name, alpha).fit(A, y[tr]).predict(B)

def med(name, tr, te, alpha=None, grouped=False):
    """median % error in Z"""
    return float(np.median(100*np.abs(pred(name, tr, te, alpha, grouped) - y[te])/D.Z.values[te]))

# ---------------------------------------------------------------- export
out = {"terms": names, "TC": TC, "RHOC": RHOC, "sources": SRC}
out["points"] = {"T": D.T_K.round(3).tolist(), "rho": (D.rho_mol_m3/1000).round(4).tolist(),   # rho in mol/L
                 "Z": D.Z.round(5).tolist(), "s": [SRC.index(s) for s in src], "near": near.astype(int).tolist()}
sc = StandardScaler().fit(X); out["scaler"] = {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist()}
Xs = sc.transform(X)

# models fitted on all usable data (shown in the explorer)
out["models"] = {}
for m in ["ridge", "bayesian ridge", "ARD"]:
    f = mk(m).fit(Xs, y)
    out["models"][m] = {"coef": f.coef_.tolist(), "b": float(f.intercept_), "n": int((np.abs(f.coef_) > 1e-8).sum())}

# lasso path (14 strengths). Index 5 is set to exactly ALPHA so the slider's default is the final model.
alphas = np.logspace(-7, -1, 14); alphas[5] = ALPHA
path = []
for a in alphas:
    f = Lasso(alpha=a, max_iter=200000).fit(Xs, y)
    loso = {s: med("lasso", src != s, src == s, a) for s in SRC}
    path.append({"alpha": float(a), "n": int((f.coef_ != 0).sum()), "coef": f.coef_.tolist(),
                 "b": float(f.intercept_), "loso": loso})
out["path"] = path
assert path[5]["n"] == 15, f"final lasso should keep 15 terms, kept {path[5]['n']}"

# leave-one-source-out table (ridge tuned with group CV inside the training sources)
out["loso"] = {m: {s: med(m, src != s, src == s, grouped=True) for s in SRC}
               for m in ["ideal gas", "virial OLS", "ridge", "bayesian ridge", "ARD"]}
out["loso"]["lasso (15 terms)"] = {s: med("lasso", src != s, src == s, ALPHA) for s in SRC}

# near-critical test: train without near-critical rows (Vennix excluded from training too), test on them
tr = (src != "Vennix 1965") & ~near; te = near
out["near"] = {"ideal gas": med("ideal gas", tr, te), "virial OLS": med("virial OLS", tr, te),
               "ridge": med("ridge", tr, te), "lasso": med("lasso", tr, te, ALPHA)}
out["n_near"] = int(near.sum())

os.makedirs("docs", exist_ok=True)
json.dump(out, open("docs/model.json", "w"), separators=(",", ":"))
print("saved docs/model.json", round(os.path.getsize("docs/model.json")/1024), "KB")

# ---------------------------------------------------------------- checks to compare with the notebook / website
kept = [n for n, c in zip(names, path[5]["coef"]) if c != 0]
print("final lasso terms (", len(kept), "):", ", ".join(kept))
print("near-critical rows:", out["n_near"], "| median % error:", {k: round(v, 2) for k, v in out["near"].items()})
print("leave-one-source-out (median % error):")
print(pd.DataFrame(out["loso"]).T.round(2).to_string())

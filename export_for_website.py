# Colab cell: needs `usable`, TC, RHOC from Step 2. Writes docs/model.json for the website.
import json, os, numpy as np, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, RidgeCV, LassoCV, BayesianRidge, ARDRegression, LinearRegression
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")
D = usable.reset_index(drop=True); y = (D.Z - 1).values; src = D.source.values
SRC = sorted(D.source.unique())
names = [f"d{k}_t{t}" for k in range(1, 7) for t in range(6)]
X = np.column_stack([D.delta.values**k * D.tau.values**t for k in range(1, 7) for t in range(6)])
Xv = np.column_stack([D.delta.values, D.delta.values*D.tau.values, D.delta.values*D.tau.values**2, D.delta.values**2, D.delta.values**2*D.tau.values])
near = (((D.T_K - TC).abs() < 10) & D.delta.between(0.6, 1.6)).values
def mk(name, alpha=None):
    if name == "ridge": return RidgeCV(alphas=np.logspace(-6, 3, 30))
    if name == "lasso": return Lasso(alpha=alpha, max_iter=200000)
    if name == "bayesian ridge": return BayesianRidge()
    if name == "ARD": return ARDRegression(max_iter=300)
def pred(name, tr, te, alpha=None):
    if name == "ideal gas": return np.zeros(te.sum())
    if name == "virial OLS": return LinearRegression(fit_intercept=False).fit(Xv[tr], y[tr]).predict(Xv[te])
    s = StandardScaler().fit(X[tr]); return mk(name, alpha).fit(s.transform(X[tr]), y[tr]).predict(s.transform(X[te]))
def med(name, tr, te, alpha=None):
    return float(np.median(100*np.abs(pred(name, tr, te, alpha) - y[te])/D.Z.values[te]))
out = {"terms": names, "TC": TC, "RHOC": RHOC, "sources": SRC}
out["points"] = {"T": D.T_K.round(3).tolist(), "rho": (D.rho_mol_m3/1000).round(4).tolist(),
                 "Z": D.Z.round(5).tolist(), "s": [SRC.index(s) for s in src], "near": near.astype(int).tolist()}
sc = StandardScaler().fit(X); out["scaler"] = {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist()}
Xs = sc.transform(X)
# fixed models fitted on all data
out["models"] = {}
for m in ["ridge", "bayesian ridge", "ARD"]:
    f = mk(m).fit(Xs, y); out["models"][m] = {"coef": f.coef_.tolist(), "b": float(f.intercept_), "n": int((np.abs(f.coef_) > 1e-8).sum())}
# lasso path, with leave-one-source-out medians
path = []
for a in np.logspace(-7, -1, 14):
    f = Lasso(alpha=a, max_iter=200000).fit(Xs, y)
    loso = {s: med("lasso", src != s, src == s, a) for s in SRC}
    path.append({"alpha": float(a), "n": int((f.coef_ != 0).sum()), "coef": f.coef_.tolist(), "b": float(f.intercept_), "loso": loso})
out["path"] = path
# leave-one-source-out table for all models
out["loso"] = {m: {s: med(m, src != s, src == s) for s in SRC} for m in ["ideal gas", "virial OLS", "ridge", "bayesian ridge", "ARD"]}
out["loso"]["lasso (15 terms)"] = {s: med("lasso", src != s, src == s, 2.0e-5) for s in SRC}
# near-critical (train without near-critical rows, Vennix held out)
tr = (src != "Vennix 1965") & ~near; te = near
out["near"] = {m: med(m, tr, te, 2.0e-5) for m in ["ideal gas", "virial OLS", "ridge", "lasso"]}
out["n_near"] = int(near.sum())
os.makedirs("docs", exist_ok=True)
json.dump(out, open("docs/model.json", "w"), separators=(",", ":"))
print("saved docs/model.json", round(os.path.getsize("docs/model.json")/1024), "KB")

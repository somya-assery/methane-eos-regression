import json, os, numpy as np, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, RidgeCV, LassoCV, BayesianRidge, ARDRegression, LinearRegression
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

# Assuming `usable`, `TC`, and `RHOC` are in the environment from earlier steps
D = usable.reset_index(drop=True)
y = (D.Z - 1).values
src = D.source.values
SRC = sorted(D.source.unique())

names = [f"d{k}_t{t}" for k in range(1, 7) for t in range(6)]
X = np.column_stack([D.delta.values**k * D.tau.values**t for k in range(1, 7) for t in range(6)])
Xv = np.column_stack([D.delta.values, D.delta.values*D.tau.values, D.delta.values*D.tau.values**2, D.delta.values**2, D.delta.values**2*D.tau.values])

near = (((D.T_K - TC).abs() < 10) & D.delta.between(0.6, 1.6)).values

def pred(name, tr, te, alpha=None):
    if name == "ideal gas": return np.zeros(te.sum())
    if name == "virial OLS": return LinearRegression(fit_intercept=False).fit(Xv[tr], y[tr]).predict(Xv[te])
    
    s = StandardScaler().fit(X[tr])
    Xtr, Xte = s.transform(X[tr]), s.transform(X[te])
    
    # Inner cross-validation on the training set only (Prevents Leakage)
    inner = list(GroupKFold(n_splits=len(np.unique(src[tr]))).split(Xtr, y[tr], src[tr]))
    
    if name == "ridge":
        m = RidgeCV(alphas=np.logspace(-6, 3, 30), cv=inner)
    elif name == "lasso CV":
        m = LassoCV(alphas=np.logspace(-7, 0, 30), cv=inner, max_iter=200000)
    elif name == "lasso":
        m = Lasso(alpha=alpha, max_iter=200000)
    elif name == "bayesian ridge":
        m = BayesianRidge()
    elif name == "ARD":
        m = ARDRegression(max_iter=300)
        
    return m.fit(Xtr, y[tr]).predict(Xte)

def med(name, tr, te, alpha=None):
    return float(np.median(100 * np.abs(pred(name, tr, te, alpha) - y[te]) / D.Z.values[te]))

out = {"terms": names, "TC": TC, "RHOC": RHOC, "sources": SRC}
out["points"] = {
    "T": D.T_K.round(3).tolist(), 
    "rho": (D.rho_mol_m3/1000).round(4).tolist(),
    "Z": D.Z.round(5).tolist(), 
    "s": [SRC.index(s) for s in src], 
    "near

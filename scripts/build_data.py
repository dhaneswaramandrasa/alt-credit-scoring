"""Synthetic telco subscribers in Jabodetabek -> WoE scorecard -> web/data/credit.json.

Everything here is simulated. The method is the one used on real operator data:
engineer behaviour features from top-ups, network events and usage, bin them,
take weight of evidence (WoE) per bin, fit a logistic regression on the WoE
values, and scale the coefficients into points. The second half runs a small
model search (the AutoML project) on the same data.

Only numpy is needed. Run from the repo root:  python scripts/build_data.py
"""
import json
import time
from pathlib import Path

import numpy as np

rng = np.random.default_rng(7)
OUT = Path(__file__).resolve().parents[1] / "web" / "data" / "credit.json"


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------- areas, hex grid
# (name, lat, lon, weight, economy). Economy shifts the latent creditworthiness of
# people living there; weight is how many subscribers live around it.
CENTRES = [
    ("Sudirman", -6.215, 106.818, 0.7, 1.2), ("Menteng", -6.196, 106.836, 0.6, 1.1),
    ("Kebayoran Baru", -6.243, 106.798, 0.8, 1.0), ("Kelapa Gading", -6.158, 106.906, 0.8, 0.8),
    ("Pantai Indah Kapuk", -6.110, 106.745, 0.5, 1.1), ("Tanah Abang", -6.188, 106.812, 0.8, 0.0),
    ("Tambora", -6.148, 106.800, 1.0, -0.8), ("Cilincing", -6.118, 106.948, 0.9, -0.9),
    ("Cakung", -6.185, 106.945, 1.0, -0.4), ("Pasar Rebo", -6.322, 106.860, 0.8, -0.1),
    ("Cengkareng", -6.150, 106.735, 1.1, -0.4), ("Bekasi Barat", -6.240, 106.990, 1.1, 0.2),
    ("Bekasi Utara", -6.205, 107.010, 1.0, -0.3), ("Cikarang", -6.300, 107.150, 1.0, -0.2),
    ("Depok Beji", -6.370, 106.825, 1.0, 0.2), ("Cimanggis", -6.380, 106.870, 0.9, 0.0),
    ("Bojonggede", -6.490, 106.795, 0.8, -0.6), ("Cibinong", -6.480, 106.850, 0.8, -0.3),
    ("Bogor Tengah", -6.595, 106.800, 0.9, 0.1), ("Tangerang Kota", -6.178, 106.632, 1.1, 0.0),
    ("BSD", -6.300, 106.660, 0.7, 1.0), ("Karawaci", -6.228, 106.610, 0.8, 0.3),
    ("Ciputat", -6.310, 106.745, 0.9, 0.2), ("Pamulang", -6.345, 106.725, 0.8, 0.0),
    ("Tigaraksa", -6.262, 106.480, 0.6, -0.8), ("Ciracas", -6.330, 106.880, 0.8, -0.2),
]
c_lat = np.array([c[1] for c in CENTRES]); c_lon = np.array([c[2] for c in CENTRES])
c_w = np.array([c[3] for c in CENTRES]); c_e = np.array([c[4] for c in CENTRES])

# pointy-top hexes about 2.2 km across
R_KM = 1.27
dy = 1.5 * R_KM / 110.6
dx = np.sqrt(3) * R_KM / 110.3
hexes = []
for r, lat in enumerate(np.arange(-6.66, -6.06, dy)):
    for lon in np.arange(106.44 + (dx / 2 if r % 2 else 0), 107.22, dx):
        d2 = ((lat - c_lat) * 110.6) ** 2 + ((lon - c_lon) * 110.3) ** 2
        k = c_w * np.exp(-d2 / (2 * 4.5 ** 2))
        dens = k.sum()
        if dens < 0.12:
            continue
        econ = (k * c_e).sum() / dens
        name = CENTRES[int(np.argmin(d2))][0]
        hexes.append((lat, lon, dens, econ, name))
hexes = np.array([(h[0], h[1], h[2], h[3]) for h in hexes]), [h[4] for h in hexes]
hx, hx_name = hexes
hx_econ = hx[:, 3] + rng.normal(0, 0.25, len(hx))
print(f"{len(hx)} hexes")

# ---------------------------------------------------------------- subscribers
N = 60_000
p = hx[:, 2] / hx[:, 2].sum()
home = rng.choice(len(hx), N, p=p)


def simulate(home, tenure_shift=0.0):
    """Latent creditworthiness z drives both the phone behaviour and default."""
    n = len(home)
    z = 0.45 * hx_econ[home] + rng.normal(0, 1, n)
    f = {
        "topup": sigmoid(0.6 + 0.75 * z + rng.normal(0, 0.6, n)),            # share of weeks with a top-up
        "tenure": np.clip(np.exp(3.1 + 0.45 * z + tenure_shift + rng.normal(0, 0.6, n)), 1, 144),  # months on network
        "nights": 30 * sigmoid(1.3 + 0.6 * z + rng.normal(0, 0.7, n)),       # nights a month at the usual home cell
        "data": np.exp(4.1 + 0.2 * z + rng.normal(0, 0.6, n)),               # data spend, IDR thousand a month
        "contacts": np.exp(3.4 + 0.25 * z + rng.normal(0, 0.6, n)),          # distinct contacts a month
        "zerobal": 30 * sigmoid(-1.9 - 0.7 * z + rng.normal(0, 0.7, n)),     # days a month at zero balance
    }
    # default within 6 months of the loan: z, things the phone never sees, and an
    # interaction (no settled home and often at zero balance) that a points table misses
    unsettled = (f["nights"] < 14) & (f["zerobal"] > 6)
    bad = rng.random(n) < sigmoid(-2.95 - 1.55 * (z + rng.normal(0, 0.3, n)) + 0.9 * unsettled)
    return f, bad.astype(int)


feats, bad = simulate(home)
print(f"bad rate {bad.mean():.3f}")

FEATURES = [
    # key, label, unit, bin edges (upper bounds; last bin open), slider range
    ("topup", "Top-up regularity", "share of weeks with a top-up", [0.3, 0.5, 0.7, 0.85], (0, 1, 0.01)),
    ("tenure", "Tenure on network", "months", [6, 12, 24, 48], (1, 120, 1)),
    ("nights", "Nights at a stable home", "nights a month", [10, 18, 24, 27], (0, 30, 1)),
    ("zerobal", "Days at zero balance", "days a month", [1, 3, 6, 10], (0, 30, 1)),
    ("contacts", "Active contacts", "people a month", [15, 30, 60], (0, 200, 1)),
    ("data", "Data spend", "IDR thousand a month", [30, 60, 120], (0, 300, 5)),
]
KEYS = [f[0] for f in FEATURES]

idx = rng.permutation(N)
tr, te = idx[: int(0.7 * N)], idx[int(0.7 * N):]


def bin_index(x, edges):
    return np.searchsorted(np.array(edges), x, side="right")


B = {k: bin_index(feats[k], e) for k, _, _, e, _ in FEATURES}


def woe_table(b, y, nb):
    g = np.bincount(b, weights=1 - y, minlength=nb) + 0.5
    s = np.bincount(b, weights=y, minlength=nb) + 0.5
    return np.log((g / g.sum()) / (s / s.sum())), g, s


woe, tables, iv = {}, {}, {}
for k, _, _, e, _ in FEATURES:
    w, g, s = woe_table(B[k][tr], bad[tr], len(e) + 1)
    woe[k] = w
    tables[k] = (g, s)
    iv[k] = float((((g / g.sum()) - (s / s.sum())) * w).sum())


def logit_fit(X, y, l2=1e-4, iters=30):
    """Logistic regression by Newton / IRLS."""
    X1 = np.c_[np.ones(len(X)), X]
    b = np.zeros(X1.shape[1])
    for _ in range(iters):
        p = sigmoid(X1 @ b)
        H = X1.T @ (X1 * (p * (1 - p))[:, None]) + l2 * np.eye(len(b))
        b -= np.linalg.solve(H, X1.T @ (p - y) + l2 * b)
    return b


def gini(y, s):
    """2*AUC - 1 for 'higher score means more likely bad'."""
    o = np.argsort(s)
    r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return 2 * auc - 1


Xw = np.c_[[woe[k][B[k]] for k in KEYS]].T
beta = logit_fit(Xw[tr], bad[tr])
a, bj = beta[0], beta[1:]

# points: 600 at odds 50:1 good:bad, 20 points to double the odds
PDO, BASE, ODDS = 20, 600, 50
factor = PDO / np.log(2)
offset = BASE - factor * np.log(ODDS)
n_f = len(KEYS)
points = {k: np.round(-(bj[j] * woe[k] + a / n_f) * factor + offset / n_f).astype(int) for j, k in enumerate(KEYS)}


def score_of(Bdict):
    return sum(points[k][Bdict[k]] for k in KEYS)


score = score_of(B)
g_tr = gini(bad[tr], -score[tr]); g_te = gini(bad[te], -score[te])
print(f"scorecard Gini train {g_tr:.3f}  holdout {g_te:.3f}")

# bands: decline the lowest ~22%, review the next ~20%
cut_decline = int(np.percentile(score, 22))
cut_approve = int(np.percentile(score, 42))
print("cutoffs", cut_decline, cut_approve)

# ---------------------------------------------------------------- distribution
lo, hi = int(score.min()) // 5 * 5, int(score.max()) // 5 * 5 + 5
edges = np.arange(lo, hi + 1, 5)
hist_g = np.histogram(score[te][bad[te] == 0], edges)[0]
hist_b = np.histogram(score[te][bad[te] == 1], edges)[0]

# ---------------------------------------------------------------- map
mean_pts = {k: float(points[k][B[k]].mean()) for k in KEYS}
hex_out = []
for h in range(len(hx)):
    m = home == h
    if m.sum() < 25:
        continue
    s = score[m]
    drag = {k: float(points[k][B[k][m]].mean() - mean_pts[k]) for k in KEYS}
    worst = min(drag, key=drag.get); best = max(drag, key=drag.get)
    hex_out.append({
        "lat": round(float(hx[h, 0]), 4), "lon": round(float(hx[h, 1]), 4), "area": hx_name[h],
        "n": int(m.sum()), "score": round(float(s.mean()), 1),
        "approve": round(float((s >= cut_approve).mean()), 3),
        "bad": round(float(bad[m].mean()), 3),
        "up": [best, round(drag[best], 1)], "down": [worst, round(drag[worst], 1)],
    })
print(f"{len(hex_out)} hexes on map")

# ---------------------------------------------------------------- monitoring (PSI)
dev_q = np.percentile(score, np.arange(10, 100, 10))


def psi(ref, cur):
    r = np.bincount(np.searchsorted(dev_q, ref), minlength=10) / len(ref) + 1e-4
    c = np.bincount(np.searchsorted(dev_q, cur), minlength=10) / len(cur) + 1e-4
    return float(((c - r) * np.log(c / r)).sum())


def csi(k, ref_b, cur_b):
    nb = len(points[k])
    r = np.bincount(ref_b, minlength=nb) / len(ref_b) + 1e-4
    c = np.bincount(cur_b, minlength=nb) / len(cur_b) + 1e-4
    return float(((c - r) * np.log(c / r)).sum())


months = []
names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
for i, mname in enumerate(names):
    # from August a SIM-bundle campaign brings in many new subscribers: shorter tenure
    new_share = 0.0 if i < 7 else min(0.2 * (i - 6), 0.7)
    hm = rng.choice(len(hx), 8000, p=p)
    f_m, _ = simulate(hm, tenure_shift=0.0)
    n_new = int(new_share * len(hm))
    if n_new:
        f_new, _ = simulate(hm[:n_new], tenure_shift=-2.6)
        for k in KEYS:
            f_m[k][:n_new] = f_new[k]
    Bm = {k: bin_index(f_m[k], e) for k, _, _, e, _ in FEATURES}
    sm = score_of(Bm)
    months.append({
        "m": mname, "psi": round(psi(score, sm), 4), "mean": round(float(sm.mean()), 1),
        "csi": {k: round(csi(k, B[k], Bm[k]), 4) for k in KEYS},
    })
print("PSI", [m["psi"] for m in months])

# ---------------------------------------------------------------- AutoML search
# Raw features, log where skewed, standardised. Same 70/30 split.
Xr = np.c_[feats["topup"], np.log(feats["tenure"]), feats["nights"], feats["zerobal"],
           np.log(feats["contacts"]), np.log(feats["data"])]
Xr = (Xr - Xr[tr].mean(0)) / Xr[tr].std(0)
Xq = np.zeros_like(Xr, dtype=np.int64)
for j in range(Xr.shape[1]):
    Xq[:, j] = np.searchsorted(np.percentile(Xr[tr, j], np.linspace(3, 97, 31)), Xr[:, j])
NB = 32


def tree_fit(Xb, g, depth, min_leaf=60, feat_frac=1.0):
    """Regression tree on pre-binned features (histogram splits). Returns nested tuples."""
    def grow(ix, d):
        val = g[ix].mean()
        if d == 0 or len(ix) < 2 * min_leaf:
            return val
        best = (0, None, None)
        tot, n = g[ix].sum(), len(ix)
        js = range(Xb.shape[1]) if feat_frac >= 1 else rng.choice(Xb.shape[1], max(1, int(feat_frac * Xb.shape[1])), replace=False)
        for j in js:
            cs = np.cumsum(np.bincount(Xb[ix, j], weights=g[ix], minlength=NB))
            cn = np.cumsum(np.bincount(Xb[ix, j], minlength=NB))
            ok = (cn >= min_leaf) & (n - cn >= min_leaf)
            if not ok.any():
                continue
            gain = np.where(ok, cs ** 2 / np.maximum(cn, 1) + (tot - cs) ** 2 / np.maximum(n - cn, 1), -1)
            t = int(np.argmax(gain))
            if gain[t] - tot ** 2 / n > best[0]:
                best = (gain[t] - tot ** 2 / n, j, t)
        if best[1] is None:
            return val
        _, j, t = best
        left = Xb[ix, j] <= t
        return (j, t, grow(ix[left], d - 1), grow(ix[~left], d - 1))
    return grow(np.arange(len(g)), depth)


def tree_predict(node, Xb):
    out = np.empty(len(Xb))
    stack = [(node, np.arange(len(Xb)))]
    while stack:
        nd, ix = stack.pop()
        if not isinstance(nd, tuple):
            out[ix] = nd
            continue
        j, t, l, r = nd
        m = Xb[ix, j] <= t
        stack += [(l, ix[m]), (r, ix[~m])]
    return out


def m_scorecard(_):
    b = logit_fit(Xw[tr], bad[tr])
    return np.c_[np.ones(len(te)), Xw[te]] @ b


def m_logit(cfg):
    b = logit_fit(Xr[tr], bad[tr], l2=cfg["l2"])
    return np.c_[np.ones(len(te)), Xr[te]] @ b


def m_tree(cfg):
    t = tree_fit(Xq[tr], bad[tr].astype(float), cfg["depth"], min_leaf=cfg.get("leaf", 60))
    return tree_predict(t, Xq[te])


def m_forest(cfg):
    pred = np.zeros(len(te))
    for _ in range(cfg["trees"]):
        s = rng.choice(tr, len(tr))
        t = tree_fit(Xq[s], bad[s].astype(float), cfg["depth"], min_leaf=40, feat_frac=0.5)
        pred += tree_predict(t, Xq[te])
    return pred


def m_gbm(cfg):
    y = bad[tr].astype(float)
    f = np.full(len(tr), np.log(y.mean() / (1 - y.mean()))); ft = np.full(len(te), f[0])
    for _ in range(cfg["rounds"]):
        t = tree_fit(Xq[tr], y - sigmoid(f), cfg["depth"], min_leaf=80)
        f += cfg["lr"] * 4 * tree_predict(t, Xq[tr]); ft += cfg["lr"] * 4 * tree_predict(t, Xq[te])
    return ft


def m_mlp(cfg):
    h, y = cfg["hidden"], bad[tr].astype(float)
    W1 = rng.normal(0, 0.5, (Xr.shape[1], h)); b1 = np.zeros(h); W2 = rng.normal(0, 0.5, h); b2 = 0.0
    P = [W1, b1, W2, np.array([b2])]; M = [np.zeros_like(q) for q in P]; V = [np.zeros_like(q) for q in P]
    for it in range(1, cfg["epochs"] + 1):
        Z = np.tanh(Xr[tr] @ P[0] + P[1]); out = sigmoid(Z @ P[2] + P[3][0])
        d = (out - y) / len(y)
        dZ = np.outer(d, P[2]) * (1 - Z ** 2)
        G = [Xr[tr].T @ dZ, dZ.sum(0), Z.T @ d, np.array([d.sum()])]
        for q in range(4):
            M[q] = 0.9 * M[q] + 0.1 * G[q]; V[q] = 0.999 * V[q] + 0.001 * G[q] ** 2
            P[q] -= 0.03 * (M[q] / (1 - 0.9 ** it)) / (np.sqrt(V[q] / (1 - 0.999 ** it)) + 1e-8)
    return np.tanh(Xr[te] @ P[0] + P[1]) @ P[2]


SEARCH = [("Scorecard (WoE + logistic)", m_scorecard, [{}])]
SEARCH += [("Logistic regression", m_logit, [{"l2": v} for v in (1e-4, 1.0, 100.0)])]
SEARCH += [("Decision tree", m_tree, [{"depth": d} for d in (2, 4, 6, 8)])]
SEARCH += [("Random forest", m_forest, [{"trees": n, "depth": d} for n in (20, 60) for d in (4, 7)])]
SEARCH += [("Gradient boosting", m_gbm, [{"rounds": n, "depth": d, "lr": lr} for n in (40, 120) for d in (2, 3) for lr in (0.05, 0.2)])]
SEARCH += [("Neural network", m_mlp, [{"hidden": h, "epochs": e} for h in (8, 32) for e in (150, 400)])]

board = []
for fam, fn, cfgs in SEARCH:
    for cfg in cfgs:
        t0 = time.perf_counter()
        s = fn(cfg)
        dt = time.perf_counter() - t0
        g = gini(bad[te], s)
        board.append({"family": fam, "cfg": cfg, "gini": round(float(g), 4), "secs": round(dt, 3)})
        print(f"  {fam:28s} {json.dumps(cfg):42s} Gini {g:.3f}  {dt:.2f}s")

# ---------------------------------------------------------------- personas
PERSONAS = [
    ("driver", "Gig driver", "Bekasi Barat", "Ride-hail driver, on the phone all day. Small top-ups almost every week.",
     {"topup": 0.88, "tenure": 38, "nights": 25, "zerobal": 2, "contacts": 130, "data": 140}),
    ("student", "University student", "Depok Beji", "Second-year student. New SIM since moving for university, tops up when the balance runs out.",
     {"topup": 0.55, "tenure": 14, "nights": 22, "zerobal": 5, "contacts": 45, "data": 95}),
    ("office", "Office worker", "Sudirman", "Admin staff at a firm on Jalan Sudirman. Same number for seven years.",
     {"topup": 0.92, "tenure": 84, "nights": 27, "zerobal": 0, "contacts": 70, "data": 110}),
    ("trader", "Market trader", "Tanah Abang", "Runs a textile stall. Calls a lot, little data, balance often hits zero before market days.",
     {"topup": 0.62, "tenure": 60, "nights": 28, "zerobal": 7, "contacts": 160, "data": 25}),
    ("arrival", "Factory newcomer", "Cikarang", "Moved from Central Java three months ago for a factory job. Still between rented rooms.",
     {"topup": 0.45, "tenure": 3, "nights": 12, "zerobal": 11, "contacts": 18, "data": 45}),
]
cent = {c[0]: (c[1], c[2]) for c in CENTRES}


def bin_labels(edges, unit_fmt):
    labs = []
    lo_ = None
    for e in edges:
        labs.append(f"< {unit_fmt(e)}" if lo_ is None else f"{unit_fmt(lo_)} to < {unit_fmt(e)}")
        lo_ = e
    labs.append(f"{unit_fmt(lo_)} or more")
    return labs


fmt = {"topup": lambda v: f"{round(v * 100)}%"}
out = {
    "meta": {"n": N, "n_test": len(te), "bad_rate": round(float(bad.mean()), 4),
             "gini_train": round(float(g_tr), 4), "gini": round(float(g_te), 4),
             "pdo": PDO, "base": BASE, "odds": ODDS, "intercept": round(float(a), 4),
             "cut_decline": cut_decline, "cut_approve": cut_approve},
    "features": [{
        "key": k, "label": lab, "unit": unit, "edges": e,
        "bins": bin_labels(e, fmt.get(k, lambda v: f"{v:g}")),
        "points": points[k].tolist(), "woe": np.round(woe[k], 3).tolist(),
        "bad_rate": np.round(tables[k][1] / (tables[k][0] + tables[k][1]), 3).tolist(),
        "share": np.round((tables[k][0] + tables[k][1]) / (tables[k][0] + tables[k][1]).sum(), 3).tolist(),
        "coef": round(float(bj[j]), 4), "iv": round(iv[k], 3), "mean_points": round(mean_pts[k], 2),
        "range": list(rng_),
    } for j, (k, lab, unit, e, rng_) in enumerate(FEATURES)],
    "dist": {"edges": edges.tolist(), "good": hist_g.tolist(), "bad": hist_b.tolist()},
    "hexes": hex_out, "hex_radius_m": R_KM * 1000,
    "months": months,
    "automl": board,
    "personas": [{"id": i, "name": nm, "area": ar, "about": ab, "lat": cent[ar][0], "lon": cent[ar][1], "x": x}
                 for i, nm, ar, ab, x in PERSONAS],
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, separators=(",", ":")))
for i, nm, ar, ab, x in PERSONAS:
    s = sum(points[k][bin_index(np.array([x[k]]), dict((f[0], f[3]) for f in FEATURES)[k])[0]] for k in KEYS)
    print(f"  persona {nm:20s} {s}")
print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")

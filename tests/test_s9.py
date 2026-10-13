import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from s9_review import min_nuclei_clocks


def toy(seed=0):
    rng = np.random.default_rng(seed); rows, X = [], []
    donors = [(f"N{i}", "natural", float(i % 3), 100) for i in range(12)] + [("Nlow", "natural", 1.0, 20), ("Nun", "natural", np.nan, 100),
                                                                            ("T1", "progestin_only", np.nan, 100), ("T2", "progestin_only", np.nan, 15)]
    for d, g, y, n in donors:
        for comp in ("stroma", "epithelium"):
            stage = 1.0 if np.isnan(y) else y
            rows.append(dict(donor=d, group=g, y=y, n=n, compartment=comp)); X.append(stage * np.linspace(0, 1, 30) + rng.normal(0, 0.05, 30))
    M = pd.DataFrame(rows); M.index = M.donor + "|" + M.compartment
    return pd.DataFrame(X, index=M.index), M


def test_min_nuclei_only_lowers_exposed_threshold():
    P, M = toy()
    hi, lo = min_nuclei_clocks(P, M, "progestin_only", 50), min_nuclei_clocks(P, M, "progestin_only", 10)
    assert "T2" not in hi.index and "T2" in lo.index and "T1" in hi.index
    assert "Nlow" not in lo.index and "Nun" in lo.index
    nat = [d for d in hi.index if hi.group[d] == "natural"]
    assert np.allclose(hi.loc[nat, ["stroma", "epithelium"]], lo.loc[nat, ["stroma", "epithelium"]])
    assert np.allclose(lo.dyssynchrony, lo.stroma - lo.epithelium)

import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from s7_external_bulk import whole_tissue


def test_whole_tissue_weights_by_nuclei():
    P = pd.DataFrame({"g": [1.0, 4.0, 2.0]}, index=["a|s", "a|e", "b|s"])
    M = pd.DataFrame({"donor": ["a", "a", "b"], "n": [3, 1, 5]}, index=P.index)
    w = whole_tissue(P, M)
    assert np.isclose(w.loc["a", "g"], (3 * 1 + 4) / 4) and np.isclose(w.loc["b", "g"], 2.0)



def test_hodges_lehmann():
    from s8_effects import hodges_lehmann
    assert hodges_lehmann([3, 4], [1, 2]) == 2.0
    assert hodges_lehmann([0, 10], [0]) == 5.0


def test_holm():
    from s9_review import holm
    adj = holm([0.01, 0.04, None, 0.03])
    assert adj[2] is None and abs(adj[0] - 0.03) < 1e-12 and abs(adj[3] - 0.06) < 1e-12 and abs(adj[1] - 0.06) < 1e-12


def test_rank_quantiles():
    import pandas as pd
    from s10_rank_bulk import rank_quantiles
    r = rank_quantiles(pd.DataFrame([[5.0, 1.0, 3.0, 2.0]]))
    assert r.iloc[0].tolist() == [1.0, 0.25, 0.75, 0.5]


def test_follow_stay():
    import pandas as pd
    from s14_pgr_cistrome import follow_stay
    G = pd.DataFrame({"compartment": ["epithelium"] * 3 + ["stroma"], "direction": ["up", "up", "down", "up"],
                      "gene": ["A", "B", "C", "D"], "fraction": [0.9, 0.1, 0.9, 0.9]})
    assert follow_stay(G, "epithelium") == ({"A"}, {"B"})


def test_marker_model_recovers_stage():
    import numpy as np
    from s15_clock_rigor import m_marker
    rng = np.random.default_rng(0)
    y = np.repeat([0.0, 1.0, 2.0], 10)
    X = rng.normal(size=(30, 200)); X[:, :50] += y[:, None]; X[:, 50:100] -= y[:, None]
    pred = m_marker(X, y)(X)
    assert np.corrcoef(pred, y)[0, 1] > 0.95

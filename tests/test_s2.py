import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from s2_clock import stage_value, treatment_group, fit, lodo


def test_labels():
    assert stage_value("Proliferative Late") == 0 and stage_value("Secretory Mid") == 2 and np.isnan(stage_value("Secretory"))
    assert treatment_group("nan") == "natural" and treatment_group("Mirena coil") == "progestin_only"
    assert treatment_group("POP - Cerezette") == "progestin_only" and treatment_group("COCP (Rigevidon)") == "combined_pill"
    assert treatment_group("norethindrone/E2:1/20") == "combined_pill" and treatment_group("Zoladex") == "other_Zoladex"


def test_lodo_recovers_a_linear_clock():
    rng = np.random.default_rng(0); y = np.repeat([0.0, 1.0, 2.0], 10)
    X = rng.normal(size=(30, 200)); X[:, :5] += y[:, None] * 2
    p = lodo(X, y)
    assert np.corrcoef(p, y)[0, 1] > 0.8

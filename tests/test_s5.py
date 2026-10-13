import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from s5_mechanism import response_fraction, label


def test_response_fraction():
    f = response_fraction([0, 1, 2], [2, 3, 0], [1, 1, 0])
    assert np.allclose(f, [0.5, 0.0, 1.0])


def test_label():
    M = pd.DataFrame({"group": ["natural", "natural", "progestin_only", "natural", "combined_pill"],
                      "Stage": ["Proliferative Late", "Secretory Mid", "Hormones", "Menstrual", "Hormones"]})
    assert label(M, "progestin_only").tolist() == ["proliferative", "secretory_mid", "treated", "other", "other"]


def test_zscore():
    from s6_external_huang import zscore
    Z = zscore([[1, 5], [3, 5]])
    assert np.allclose(Z, [[-1, 0], [1, 0]])

import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from s4_sensitivity import residuals, perm_p, drug


def test_residuals_on_a_line():
    x = np.arange(6.0); y = 2 * x + 1
    r_nat, r_new = residuals(x, y, (np.array([1.0, 2.0]), np.array([3.0, 4.0])))
    assert np.allclose(r_nat, 0)
    assert np.allclose(r_new, [0.0, -1.0])


def test_perm_p_directions():
    a, b = np.full(5, 3.0) + np.arange(5) * 0.01, np.zeros(20) + np.arange(20) * 0.01
    assert perm_p(a, b, n=2000) < 0.01
    assert perm_p(a, b, n=2000, less=True) > 0.9


def test_drug():
    assert drug("Mirena coil") == "LNG-IUS" and drug("POP - Cerezette") == "desogestrel" and drug("Implant") == "implant"

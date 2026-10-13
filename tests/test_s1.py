import os, sys
import numpy as np, scipy.sparse as sp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from s1_profiles import compartment, lognorm


def test_compartment_rules():
    assert compartment("Mesenchymal", "dStromal_mid") == "stroma" and compartment("Mesenchymal", "ePV_2") is None
    assert compartment("Mesenchymal", "uSMCs") is None and compartment("Epithelial", "eHormones") == "epithelium"
    assert compartment("Immune", "Immune_Myeloid") == "immune" and compartment("Endothelial", "Venous") == "endothelium"


def test_lognorm_scales_each_cell_to_10k():
    Y = lognorm(sp.csr_matrix(np.array([[1.0, 3.0], [0.0, 2.0]])))
    assert np.allclose(Y.toarray(), np.log1p(np.array([[2500, 7500], [0, 10000]])))

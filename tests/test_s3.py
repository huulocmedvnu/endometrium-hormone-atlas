import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from s3_endometriosis_age import stage_value, partial_spearman, stouffer


def test_stage_and_stats():
    assert stage_value("Secretory Early-Mid") == 1.5 and stage_value("Proliferative Early") == 0 and np.isnan(stage_value("Menstrual"))
    rng = np.random.default_rng(0); z = rng.normal(size=200); x = z + rng.normal(size=200) * 0.1; y = z + rng.normal(size=200) * 0.1
    assert partial_spearman(x, y, z)[0] < 0.3                         # x and y only share z
    assert stouffer([0.01, 0.01]) < 0.01 and abs(stouffer([0.5, 0.5]) - 0.5) < 1e-9

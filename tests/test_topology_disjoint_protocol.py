import numpy as np

from scripts.run_metr_la_topology_disjoint import (
    DEV_GROUPS,
    TEST_GROUPS,
    MASK_REPS,
    OBS_RATIOS,
    all_offdiagonal_support,
    make_candidate_masks,
)


def test_topology_disjoint_group_roles_and_scenario_count():
    assert set(DEV_GROUPS).isdisjoint(TEST_GROUPS)
    assert len(DEV_GROUPS) == 4
    assert len(TEST_GROUPS) == 6
    assert len(TEST_GROUPS) * len(MASK_REPS) * len(OBS_RATIOS) == 24


def test_all_offdiagonal_candidate_support():
    support = all_offdiagonal_support(20)
    assert support.shape == (20, 20)
    assert int(support.sum()) == 380
    assert not np.any(np.diag(support))


def test_candidate_masks_partition_and_nesting():
    seed = 310100
    observed40, fit40, validation40, missing40 = make_candidate_masks(20, seed, 0.4, 0.2)
    observed70, fit70, validation70, missing70 = make_candidate_masks(20, seed, 0.7, 0.2)
    support = all_offdiagonal_support(20)
    assert int(observed40.sum()) == 152
    assert int(observed70.sum()) == 266
    assert np.all(observed40 <= observed70)
    for observed, fit, validation, missing in [
        (observed40, fit40, validation40, missing40),
        (observed70, fit70, validation70, missing70),
    ]:
        assert not np.any(fit & validation)
        assert not np.any(fit & missing)
        assert not np.any(validation & missing)
        assert np.array_equal(fit | validation, observed)
        assert np.array_equal(observed | missing, support)

import json

import yaml

from scripts.run_metr_la_topology_disjoint import (
    DEV_GROUPS,
    TEST_GROUPS,
    MASK_REPS,
    OBS_RATIOS,
    LAMBDA_GRID,
)
from src.utils import REPO_ROOT


PAPER_TITLE = (
    'Aggregation-Aware In-Strength Calibration for Information Fusion '
    'over Incomplete Directed Graphs'
)


def test_release_title_is_consistent():
    readme = (REPO_ROOT / 'README.md').read_text(encoding='utf-8')
    citation = (REPO_ROOT / 'CITATION.bib').read_text(encoding='utf-8')
    assert readme.startswith(f'# {PAPER_TITLE}\n')
    assert PAPER_TITLE in citation


def test_metr_config_matches_node_disjoint_protocol():
    cfg = yaml.safe_load(
        (REPO_ROOT / 'configs/metr_la.yaml').read_text(encoding='utf-8')
    )
    assert cfg['node_split']['development_group_ids'] == DEV_GROUPS
    assert cfg['node_split']['test_group_ids'] == TEST_GROUPS
    assert cfg['observation_ratios'] == OBS_RATIOS
    assert cfg['mask_repetitions'] == MASK_REPS
    assert cfg['candidate_relations']['definition'] == 'all_directed_nonself_pairs'
    assert int(cfg['candidate_relations']['directed_pair_count']) == 380
    assert [float(v) for v in cfg['ridge']['lambda_grid']] == LAMBDA_GRID
    assert float(cfg['ridge']['selected_lambda']) == 100.0


def test_protocol_manifest_matches_release():
    manifest = json.loads(
        (
            REPO_ROOT
            / 'results/frozen/metr_la_topology_disjoint/protocol_manifest.json'
        ).read_text(encoding='utf-8')
    )
    assert manifest['uses_adjacency_for_split'] is False
    assert manifest['candidate_rule_uses_truth'] is False
    assert int(manifest['development_graphs']) == 4
    assert int(manifest['test_graphs']) == 6
    assert int(manifest['candidate_rule'].split('=')[-1].rstrip(')')) == 380
    assert float(manifest['selected_lambda']) == 100.0
    assert manifest['raw_asgc_share_same_completion'] is True

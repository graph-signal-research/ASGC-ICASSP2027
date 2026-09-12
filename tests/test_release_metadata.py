import json
from pathlib import Path

import numpy as np
import yaml

from src.data.metr_la import candidate_relation_support
from src.experiments.metr_la import load_checkpoint
from src.utils import REPO_ROOT, sha256_file


PAPER_TITLE = (
    'Aggregation-Aware In-Strength Calibration for Information Fusion '
    'over Incomplete Directed Graphs'
)


def test_release_title_is_consistent():
    readme = (REPO_ROOT / 'README.md').read_text(encoding='utf-8')
    citation = (REPO_ROOT / 'CITATION.bib').read_text(encoding='utf-8')
    assert readme.startswith(f'# {PAPER_TITLE}\n')
    assert PAPER_TITLE in citation


def test_candidate_relation_support_definition():
    adjacency = np.array(
        [
            [0.0, 0.2, 0.0],
            [0.4, 0.0, 0.3],
            [0.0, 0.0, 0.0],
        ]
    )
    support = candidate_relation_support(adjacency, expected_edge_count=3)
    assert support.sum() == 3
    assert not np.any(np.diag(support))
    assert support[0, 1] and support[1, 0] and support[1, 2]


def test_frozen_subgraph_support_matches_config():
    cfg = yaml.safe_load(
        (REPO_ROOT / 'configs/metr_la.yaml').read_text(encoding='utf-8')
    )
    manifest = json.loads(
        (
            REPO_ROOT
            / 'results/frozen/metr_la/development/subgraph_manifest.json'
        ).read_text(encoding='utf-8')
    )
    assert cfg['relation_support']['definition'] == 'nonzero_offdiagonal_entries'
    assert int(cfg['relation_support']['directed_edge_count']) == 147
    assert int(manifest['directed_nonzero_nonself_edges']) == 147


def test_released_checkpoint_loads_with_config():
    cfg = yaml.safe_load(
        (REPO_ROOT / 'configs/metr_la.yaml').read_text(encoding='utf-8')
    )
    checkpoint = (
        REPO_ROOT
        / 'results/frozen/metr_la/heldout_masks/models/gat_seed_2027106_r40.pt'
    )
    model = load_checkpoint(checkpoint, cfg['gat'])
    assert model.linear.in_features == int(cfg['gat']['input_dim'])


def test_checkpoint_manifest_hashes():
    manifest = json.loads(
        (REPO_ROOT / 'docs/CHECKPOINT_MANIFEST.json').read_text(
            encoding='utf-8'
        )
    )
    assert len(manifest) == 20
    for item in manifest:
        assert sha256_file(REPO_ROOT / item['file']) == item['sha256']

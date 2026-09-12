from pathlib import Path
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import shutil

import requests

from src.utils import load_config

FILES = {
    'graph_sensor_ids.txt': (
        'https://raw.githubusercontent.com/liyaguang/DCRNN/master/'
        'data/sensor_graph/graph_sensor_ids.txt'
    ),
    'distances_la_2012.csv': (
        'https://raw.githubusercontent.com/liyaguang/DCRNN/master/'
        'data/sensor_graph/distances_la_2012.csv'
    ),
    'graph_sensor_locations.csv': (
        'https://raw.githubusercontent.com/liyaguang/DCRNN/master/'
        'data/sensor_graph/graph_sensor_locations.csv'
    ),
    'adj_mx.pkl': (
        'https://raw.githubusercontent.com/liyaguang/DCRNN/master/'
        'data/sensor_graph/adj_mx.pkl'
    ),
}

EXPECTED_SHA256 = {
    'adj_mx.pkl': 'a35687c6e15aa228dc45027b0ed2a0ea0f4ec78f573deb992c595092d12f61b3',
    'distances_la_2012.csv': 'a576a2a3e28dbb959be6da22688e24dd1b246b81264595e129147c256cd53de5',
    'graph_sensor_ids.txt': '3ba026caa2e6263ab0ea54b0fa1b125dbfa7216544cd05313b555e826292b990',
    'graph_sensor_locations.csv': 'eb8ea96e07358b45d0e4ba3b89c2673fa20c54af50150249e627389e749ade6f',
}


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare the METR-LA data files.')
    parser.add_argument('--config', default='configs/metr_la.yaml')
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Destination directory; defaults to the config data_dir.',
    )
    parser.add_argument(
        '--metr-h5',
        type=Path,
        help='Path to a user-downloaded metr-la.h5 file.',
    )
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = ROOT / (args.output_dir or Path(cfg['data_dir']))
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        print('download script smoke test: PASS; no network request made')
        return

    for name, url in FILES.items():
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        digest = _sha256_bytes(response.content)
        expected = EXPECTED_SHA256[name]
        if digest != expected:
            raise RuntimeError(
                f'Unexpected SHA-256 for {name}: {digest}; expected {expected}'
            )
        (output_dir / name).write_bytes(response.content)
        print(f'downloaded and verified {name}')

    if args.metr_h5:
        shutil.copy2(args.metr_h5, output_dir / 'metr-la.h5')
        print('copied metr-la.h5')
    else:
        print(
            'metr-la.h5 is not redistributed. Download it from the DCRNN data '
            'release and rerun with --metr-h5 /path/to/metr-la.h5.'
        )


if __name__ == '__main__':
    main()

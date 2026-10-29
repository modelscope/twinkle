# Copyright (c) ModelScope Contributors. All rights reserved.
"""Download an explicitly selected Open-Jev split as local JSONL."""
import argparse
import json
from pathlib import Path


def main():
    from modelscope.msdatasets import MsDataset
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['ZefanCai/Open-Jev', 'ZefanCai/Open-Jev-v1.1'],
                        default='ZefanCai/Open-Jev')
    parser.add_argument('--revision', required=True)
    parser.add_argument('--subset', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    dataset = MsDataset.load(args.dataset, subset_name=args.subset, split=args.split,
                             version=args.revision).to_hf_dataset()
    with output.open('x') as stream:
        for row in dataset:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps({'rows': len(dataset), **vars(args)}))


if __name__ == '__main__':
    main()

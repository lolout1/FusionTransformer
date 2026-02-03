#!/usr/bin/env python3
"""Summarize training results from multiple experiments."""
import pickle
import statistics
import sys
from pathlib import Path

DATASET_NAMES = {
    'smartfallmm': 'SmartFallMM',
    'upfall': 'UP-FALL',
    'wedafall': 'WEDA-FALL',
    'train_all_smartfallmm': 'SmartFallMM',
    'train_all_upfall': 'UP-FALL',
    'train_all_wedafall': 'WEDA-FALL'
}

def summarize(work_dir: Path) -> dict:
    pkl = work_dir / 'fold_results.pkl'
    if not pkl.exists():
        return None
    data = pickle.load(open(pkl, 'rb'))
    f1s = [r['test']['f1_score'] for r in data]
    accs = [r['test']['accuracy'] for r in data]
    return {
        'f1': statistics.mean(f1s),
        'f1_std': statistics.stdev(f1s) if len(f1s) > 1 else 0,
        'acc': statistics.mean(accs),
        'folds': len(f1s)
    }

def get_dataset_name(path: Path) -> str:
    name = path.name
    for key, display in DATASET_NAMES.items():
        if key in name.lower():
            return display
    return name

def main():
    dirs = [Path(d) for d in sys.argv[1:]] if len(sys.argv) > 1 else []

    print(f"{'Dataset':<15} {'F1 (%)':<20} {'Accuracy (%)':<15} {'Folds'}")
    print("-" * 60)

    for d in dirs:
        name = get_dataset_name(d)
        result = summarize(d)
        if result:
            print(f"{name:<15} {result['f1']:.2f} ± {result['f1_std']:.2f}         {result['acc']:.2f}            {result['folds']}")
        else:
            print(f"{name:<15} No results")

if __name__ == '__main__':
    main()

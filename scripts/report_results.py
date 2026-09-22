"""Verify aggregate historical metrics and render confusion matrices as SVG."""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'generated')
    args = parser.parse_args()
    results = json.loads((ROOT / 'reference_results/public_test_ensemble.json').read_text())['models']
    titles = {'ce_vit_reference': 'CE-only ViT', 'triplet_vit': 'Triplet-ViT',
              'notebook_efficientnet': 'EfficientNetV2-S',
              'triplet_vit_notebook_efficientnet_50_50': '50:50 Ensemble'}
    summary = {}
    for key, result in results.items():
        cm = result['confusion_matrix']
        assert sum(map(sum, cm)) == 300 and all(sum(row) == 100 for row in cm)
        tp = cm[2][2]
        fp = cm[0][2] + cm[1][2]
        f1 = [2 * cm[i][i] / (sum(cm[i]) + sum(row[i] for row in cm)) for i in range(3)]
        calculated = {'accuracy': sum(cm[i][i] for i in range(3)) / 300,
            'macro_f1': sum(f1) / 3, 'c_recall': tp / 100,
            'c_precision': tp / (tp + fp), 'c_specificity': (200 - fp) / 200}
        for metric, value in calculated.items():
            assert math.isclose(value, result[metric], abs_tol=1e-12), (key, metric)
        assert math.isclose(f1[2], result['per_class_f1']['C'], abs_tol=1e-12)
        summary[titles[key]] = {**calculated, 'c_f1': f1[2], 'c_to_a': cm[2][0], 'c_to_b': cm[2][1]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'public_test_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1060" height="320" viewBox="0 0 1060 320">',
             '<rect width="1060" height="320" fill="white"/>',
             '<g font-family="Arial,sans-serif" text-anchor="middle" font-size="13">']
    for panel, key in enumerate(list(titles)[1:]):
        x = 50 + panel * 350
        parts.append(f'<text x="{x+120}" y="25">{titles[key]} (n=300)</text>')
        parts.append(f'<text x="{x+120}" y="46">Predicted category</text>')
        for i in range(3):
            parts.append(f'<text x="{x+40+i*80}" y="65">{"ABC"[i]}</text>')
            parts.append(f'<text x="{x-13}" y="{118+i*80}">{"ABC"[i]}</text>')
            for j in range(3):
                n = results[key]['confusion_matrix'][i][j]
                tone = int(245 - n * 1.65)
                parts.append(f'<rect x="{x+j*80}" y="{75+i*80}" width="80" height="80" fill="rgb({tone},{tone},245)" stroke="white"/>')
                parts.append(f'<text x="{x+40+j*80}" y="{120+i*80}">{n}</text>')
    parts.extend(['</g>', '</svg>'])
    (args.output / 'public_test_confusion_matrices.svg').write_text('\n'.join(parts))
    print('PASS: all four public-test metric sets recomputed from confusion matrices.')
    print(f'Wrote summaries and SVG to {args.output}')

if __name__ == '__main__':
    main()

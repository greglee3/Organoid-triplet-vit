"""Plot ROC curves from newly generated public-test evaluation arrays."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/05_final_tuning/PROFESSOR_FEEDBACK'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vit-results', type=Path, default=RUN / '07_locked_triplet_final_test/final_public_test')
    parser.add_argument('--ensemble-results', type=Path, default=RUN / '11_locked_triplet_notebook_eff_final_test/final_public_test')
    parser.add_argument('--output', type=Path, default=ROOT / 'generated')
    args = parser.parse_args()
    labels = np.load(args.vit_results / 'test_labels.npy', allow_pickle=False)
    assert np.array_equal(labels, np.load(args.ensemble_results / 'test_labels.npy', allow_pickle=False))
    paths = [('Triplet-ViT', args.vit_results / 'triplet_test_probabilities.npy'),
             ('EfficientNetV2-S', args.ensemble_results / 'efficient_test_probabilities.npy'),
             ('50:50 Ensemble', args.ensemble_results / 'triplet_eff_50_50_test_probabilities.npy')]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for title, path in paths:
        probs = np.load(path, allow_pickle=False)
        assert probs.shape == (len(labels), 3) and np.isfinite(probs).all()
        assert np.allclose(probs.sum(axis=1), 1, atol=1e-5)
        for i, ax in enumerate(axes):
            binary = labels == i
            x, y, _ = roc_curve(binary, probs[:, i])
            auc = roc_auc_score(binary, probs[:, i])
            ax.plot(x, y, label=f'{title} ({auc:.4f})', linewidth=1)
            print(f'{title} {"ABC"[i]} AUC={auc:.5f}')
    for i, ax in enumerate(axes):
        ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=.8)
        ax.set(title=f'Category {"ABC"[i]}', xlabel='False positive rate', ylabel='True positive rate', xlim=(0,1), ylim=(0,1))
        ax.legend(loc='lower right', fontsize=7)
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / 'public_test_roc.png', dpi=300)
    fig.savefig(args.output / 'public_test_roc.svg')
    plt.close(fig)

if __name__ == '__main__':
    main()

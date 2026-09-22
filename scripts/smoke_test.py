"""CPU-only tests on synthetic tensors; no downloads, data or training loop."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '05_final_tuning/02_ordinal_aware_triplet'))
from train_pilot import PKBatchSampler, ViTSharedMetricModel, ordinal_semi_hard_triplet_loss

def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    labels = np.repeat(np.arange(3), 20)
    batches = list(PKBatchSampler(labels, 3, 8, 42))
    for batch in batches:
        assert len(batch) == 24 and len(set(batch)) == 24
        assert np.array_equal(np.bincount(labels[batch]), [8, 8, 8])
    assert batches == list(PKBatchSampler(labels, 3, 8, 42))
    angles = torch.deg2rad(torch.tensor([0., 5., 15., 20., 60., 65.]))
    z = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1).requires_grad_()
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    loss, dp, dn, count = ordinal_semi_hard_triplet_loss(z, y, 0.1, 0.1)
    assert count > 0 and 0 < float(loss) < 0.1 and dp < dn
    loss.backward()
    assert torch.isfinite(z.grad).all()
    permuted = ordinal_semi_hard_triplet_loss(z.detach(), torch.tensor([0,0,2,2,1,1]), .1, .1)
    assert torch.allclose(loss, permuted[0]) and count == permuted[3]
    zero = ordinal_semi_hard_triplet_loss(torch.eye(3), torch.arange(3), .1, .1)
    assert float(zero[0]) == 0 and zero[3] == 0
    print('PASS: balanced sampling, uniform semi-hard loss, gradients, empty-triplet case')

    model = ViTSharedMetricModel(pretrained=False).eval()
    with torch.no_grad():
        logits, embeddings = model(torch.zeros(1, 3, 224, 224))
    assert logits.shape == (1, 3) and embeddings.shape == (1, 384)
    assert model.metric_projection is None
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(1), atol=1e-5)
    del model
    path = ROOT / '05_final_tuning/PROFESSOR_FEEDBACK/03_vit_triplet_efficientnet/train_efficientnet_oof.py'
    spec = importlib.util.spec_from_file_location('efficient_smoke', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.create_model(pretrained=False, trainable_mode='all').eval()
    assert all(p.requires_grad for p in model.parameters())
    with torch.no_grad():
        assert model(torch.zeros(1, 3, 224, 224)).shape == (1, 3)
    print('PASS: ViT 384-D normalized embeddings and EfficientNet 3-class forward passes')

if __name__ == '__main__':
    main()

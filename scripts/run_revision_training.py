"""Run the published revision recipes; dry-run unless --execute is supplied.

This trains new models, not a bitwise replay of historical GPU results. Only use
trusted checkpoints produced by these trainers: the historical evaluator loads
PyTorch checkpoint objects. The original ZIP member names are required for exact
identity matching. No historical seed-42 outputs or locks are modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PF = ROOT / "05_final_tuning/PROFESSOR_FEEDBACK"
VIT = ROOT / "05_final_tuning/02_ordinal_aware_triplet/train_pilot.py"
EFF = PF / "03_vit_triplet_efficientnet/train_efficientnet_oof.py"
EVALUATOR = PF / "07_locked_triplet_final_test/evaluate_locked_test_once.py"
SEEDS = (70384, 12731, 76313, 98831, 80324, 104729, 130363, 155921, 180749, 205019)
MODELS = {"ce": "CE-only ViT", "triplet": "Triplet-ViT",
          "efficientnet": "EfficientNetV2-S", "ensemble": "50:50 ensemble"}
CONDITIONS = (("ce", "lambda_0", 0.0), ("triplet", "lambda_0p0025", 0.0025))
VIT_SETTINGS = {
    "ordinal_lambda": 0, "lambdas": "0,0.0025", "mining_mode": "semi-hard",
    "adjacent_margin": 0.1, "far_margin": 0.1, "embedding_head": "shared",
    "projection_dim": 384, "distance_metric": "cosine", "embedding_normalization": "l2",
    "epochs": 20, "folds": 5, "classes_per_batch": 3, "samples_per_class": 8,
    "workers": 8, "lr": 3e-5, "head_lr_multiplier": 10, "optimizer": "adamw",
    "weight_decay": 0.01, "warmup_epochs": 2, "scheduler": "cosine",
    "scheduler_milestones": "10,15", "scheduler_gamma": 0.1,
    "triplet_ce_only_epochs": 0, "triplet_ramp_epochs": 0, "augmentation": "paper",
    "min_c_specificity": 0.90, "max_c_recall_drop": 0.02, "max_c_precision_drop": 0.03,
}
EFF_SETTINGS = {
    "epochs": 50, "batch_size": 35, "workers": 16, "lr": 0.00015,
    "backbone_lr": 0.00015, "weight_decay": 0.01, "trainable_mode": "all",
    "scheduler": "step", "step_size": 3, "gamma": 0.97,
    "gridmask_probability": 0.3, "gridmask_implementation": "notebook",
    "horizontal_flip_probability": 0.5, "vertical_flip_probability": 0.5,
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    # Exclusive creation prevents silently replacing a prior result.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False,
                  default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v))
        stream.write("\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes():
    files = [VIT, EFF, EVALUATOR, Path(__file__).resolve(), ROOT / "asano_data.py",
             ROOT / "training_common.py", ROOT / "paper_reproduction.py",
             VIT.parent / "projection_head.py", VIT.parent / "triplet_schedule.py"]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}


def fold_path(seed):
    return ROOT / f"revision/folds/seed_{seed}.json"


def run_path(args, seed, stage):
    return args.runs_root / f"seed_{seed}" / stage


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("vit", "efficientnet", "evaluate", "all"), default="all")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)),
                        help="Comma-separated subset of the ten published seeds; no duplicates")
    parser.add_argument("--data", type=Path, default=ROOT / "data/Asano_Full_Data.zip",
                        help="Original dataset ZIP with published member names")
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs/revision")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--execute", action="store_true", help="Actually train or access test images")
    args = parser.parse_args()
    try:
        seeds = [int(x.strip()) for x in args.seeds.split(",")]
    except ValueError:
        parser.error("--seeds must be comma-separated integers")
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds).issubset(SEEDS):
        parser.error(f"--seeds must be a unique subset of {SEEDS}")
    args.seeds = [seed for seed in SEEDS if seed in seeds]
    args.data = args.data.expanduser().resolve()
    args.runs_root = args.runs_root.expanduser().resolve()
    if args.runs_root == ROOT or args.runs_root in ROOT.parents:
        parser.error("--runs-root must not be the repository root or its ancestor")
    return args


def command(args, seed, stage):
    script, settings = (VIT, VIT_SETTINGS) if stage == "vit" else (EFF, EFF_SETTINGS)
    cmd = [sys.executable, "-u", str(script), "--data", str(args.data),
           "--output", str(run_path(args, seed, stage)), "--seed", str(seed),
           "--device", args.device]
    if stage == "efficientnet":
        cmd += ["--folds-json", str(fold_path(seed))]
    for key, value in settings.items():
        cmd += ["--" + key.replace("_", "-"), str(value)]
    return cmd


def same_folds(actual, expected):
    # Do not compare irrelevant source-path metadata.
    for key in ("seed", "ordered_samples", "folds"):
        require(actual[key] == expected[key], f"Fold manifest mismatch: {key}")


def validate_training_identities(args):
    import numpy as np
    from sklearn.model_selection import KFold
    from asano_data import AsanoDataset
    require(args.data.is_file() and args.data.suffix.lower() == ".zip",
            "Use the original dataset ZIP; extracted-directory identifiers differ")
    dataset = AsanoDataset(args.data, "train")
    require(len(dataset) == 1200 and np.bincount(dataset.targets).tolist() == [400, 400, 400],
            "Expected 1,200 training images, 400 per category")
    ordered = [dict(index=i, sample=p, label=int(y)) for i, (p, y) in enumerate(dataset.samples)]
    manifests = {}
    for seed in args.seeds:
        published = read_json(fold_path(seed))
        generated = {"seed": seed, "ordered_samples": ordered, "folds": [
            {"fold": i, "train_indices": a.tolist(), "validation_indices": b.tolist()}
            for i, (a, b) in enumerate(KFold(5, shuffle=True, random_state=seed).split(np.arange(1200)), 1)]}
        same_folds(generated, published)
        manifests[seed] = published
    return manifests


def vit_config(seed):
    return dict(seed=seed, final_folds=5, epochs=20, lambda_ordinal=0.0,
                mining_mode="semi-hard", adjacent_margin=0.1, far_margin=0.1,
                classes_per_batch=3, samples_per_class=8, backbone_lr=3e-5,
                head_lr_multiplier=10, optimizer="adamw", weight_decay=0.01,
                warmup_epochs=2, scheduler="cosine", augmentation="paper",
                embedding_head="shared", triplet_ce_only_epochs=0, triplet_ramp_epochs=0)


def check_indices(checkpoint, split):
    require(int(checkpoint["fold"]) == split["fold"], "Checkpoint fold number differs")
    for key in ("train_indices", "validation_indices"):
        require(checkpoint[key] == split[key], f"Checkpoint {key} differs from published fold")


def validate_checkpoints(args, manifests, ev, torch):
    locks = []
    for seed in args.seeds:
        vit_root = run_path(args, seed, "vit")
        same_folds(read_json(vit_root / "folds.json"), manifests[seed])
        for stage, condition in (("vit", "lambda_0"), ("vit", "lambda_0p0025"), ("efficientnet", "")):
            folder = run_path(args, seed, stage) / condition
            require((folder / "oof_metrics.json").is_file(), f"Incomplete training: {folder}")
            for split in manifests[seed]["folds"]:
                path = folder / f"folds/fold_{split['fold']}_best.pt"
                ck = torch.load(path, map_location="cpu", weights_only=False)
                check_indices(ck, split)
                if stage == "vit":
                    ev.validate_checkpoint(ck, vit_config(seed), 0.0 if condition == "lambda_0" else 0.0025, split["fold"])
                    require(list(ck["class_names"]) == list("ABC"), "ViT class order differs")
                    for key in ("distance_metric", "embedding_normalization", "projection_dim"):
                        require(ck["args"][key] == VIT_SETTINGS[key], f"ViT setting differs: {key}")
                else:
                    require(ck["seed"] == seed, "EfficientNet checkpoint seed differs")
                    for key, value in EFF_SETTINGS.items():
                        require(ck["args"][key] == value, f"EfficientNet setting differs: {key}")
                require(not ck["args"].get("no_pretrained", False), "Expected pretrained initialization")
                require(ck["args"].get("max_batches") is None and ck["args"].get("max_folds") is None,
                        "Refusing smoke-test/partial-run checkpoints")
                locks.append(dict(seed=seed, stage=stage, condition=condition, fold=split["fold"],
                                  path=path.relative_to(args.runs_root).as_posix(), sha256=sha256(path)))
                del ck
    return locks


def evaluate(args, manifests):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    ev = load_module("revision_original_evaluator", EVALUATOR)
    trainer = ev.load_trainer()
    efficient = load_module("revision_original_efficientnet", EFF)
    # Validate every selected checkpoint and ordered training identity before
    # constructing any test dataset. No seed-42 hash/lock is repurposed.
    locks = validate_checkpoints(args, manifests, ev, torch)
    out = args.runs_root / "evaluation"
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "CHECKPOINT_LOCK.json", locks)
    write_json(out / "PROTOCOL.json", dict(seeds=args.seeds, device=args.device,
               source_sha256=source_hashes(),
               fold_sha256={str(s): sha256(fold_path(s)) for s in args.seeds},
               python_version=sys.version,
               packages={name: version(name) for name in ("torch", "torchvision", "timm", "numpy", "scikit-learn")},
               aggregation="Average five softmax vectors per architecture within seed; ensemble is 0.5 Triplet + 0.5 EfficientNet; never average across seeds",
               test_used_for_tuning=False))
    tf = ev.transforms.Compose([ev.transforms.Resize((224, 224), antialias=True), ev.transforms.ToTensor()])
    dataset = ev.AsanoDataset(args.data, "test", transform=tf)
    other = efficient.AsanoDataset(args.data, "test", transform=tf)
    require(dataset.samples == other.samples, "ViT and EfficientNet test identities differ")
    labels = np.asarray(dataset.targets)
    require(len(dataset) == 300 and np.bincount(labels).tolist() == [100, 100, 100],
            "Expected the same balanced 300-image test set")
    sample_manifest = [dict(index=i, path=p, label=int(y)) for i, (p, y) in enumerate(dataset.samples)]
    published_test = ROOT / "revision/results/test_sample_manifest.json"
    require(published_test.is_file(), f"Missing published test manifest: {published_test}")
    require(sample_manifest == read_json(published_test), "Published test identities differ")
    write_json(out / "TEST_SAMPLE_MANIFEST.json", sample_manifest)
    np.save(out / "test_labels.npy", labels)
    loader = DataLoader(dataset, batch_size=25, shuffle=False, num_workers=4)
    device = torch.device(args.device)
    rows = []
    with (out / "test_predictions.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed", "model", "sample", "true_label", "prob_A", "prob_B", "prob_C", "index", "prediction"])
        for seed in args.seeds:
            trainer.set_deterministic(seed)
            probabilities = {}
            for slug, condition, weight in CONDITIONS:
                _, p, y = ev.evaluate_condition(trainer, run_path(args, seed, "vit"), condition,
                                                 weight, vit_config(seed), loader, device)
                require(np.array_equal(y, labels), "ViT test label order differs")
                probabilities[slug] = p
            fold_probabilities = []
            for fold in range(1, 6):
                ck = torch.load(run_path(args, seed, "efficientnet") / f"folds/fold_{fold}_best.pt",
                                map_location="cpu", weights_only=False)
                model = efficient.create_model(pretrained=False, trainable_mode="all").to(device)
                model.load_state_dict(ck["model"])
                p, y = efficient.predict(model, loader, device, None)
                require(np.array_equal(y, labels), "EfficientNet test label order differs")
                fold_probabilities.append(p)
                del model, ck
                if args.device == "cuda":
                    torch.cuda.empty_cache()
            probabilities["efficientnet"] = np.stack(fold_probabilities).mean(0)
            probabilities["ensemble"] = (probabilities["triplet"] + probabilities["efficientnet"]) / 2
            folder = out / f"test/seed_{seed}"
            folder.mkdir(parents=True, exist_ok=False)
            for slug, name in MODELS.items():
                p = probabilities[slug]
                require(p.shape == (300, 3) and np.isfinite(p).all() and np.allclose(p.sum(1), 1, atol=1e-5)
                        and np.all((p >= 0) & (p <= 1)), "Invalid test probabilities")
                np.save(folder / f"{slug}_probabilities.npy", p)
                metrics = ev.extended_metrics(trainer, labels, p)
                metrics["c_f1"] = metrics["per_class_f1"]["C"]
                rows.append(dict(seed=seed, model=name, metrics=metrics))
                for i, (sample, y) in enumerate(dataset.samples):
                    writer.writerow([seed, name, sample, "ABC"[y], *p[i].tolist(), i, "ABC"[p[i].argmax()]])
            print(f"Evaluated seed {seed}: four models on the same 300 images", flush=True)
    summary = {}
    for name in MODELS.values():
        summary[name] = {}
        for key in ("accuracy", "macro_f1", "c_recall", "c_precision", "c_specificity", "c_f1"):
            values = np.array([r["metrics"][key] * 100 for r in rows if r["model"] == name])
            summary[name][key] = dict(mean=float(values.mean()), sd=float(values.std(ddof=1)) if len(values) > 1 else None)
    write_json(out / "RESULTS.json", dict(seeds=args.seeds, rows=rows, summary=summary,
               note="Metrics calculated per seed after five-fold averaging; sample SD across seeds; no cross-seed probability averaging. SD is null for one seed."))
    write_json(out / "COMPLETE.json", dict(seeds=args.seeds, test_images=300))


def main():
    args = parse_args()
    stages = [s for s in ("vit", "efficientnet") if args.stage in (s, "all")]
    for stage in stages:
        for seed in args.seeds:
            cmd = command(args, seed, stage)
            print(subprocess.list2cmdline(cmd) if os.name == "nt" else shlex.join(cmd))
    if args.stage in ("evaluate", "all"):
        evaluation_cmd = [sys.executable, str(Path(__file__).resolve()), "--stage", "evaluate",
                          "--seeds", ",".join(map(str, args.seeds)), "--data", str(args.data),
                          "--runs-root", str(args.runs_root), "--device", args.device, "--execute"]
        print(subprocess.list2cmdline(evaluation_cmd) if os.name == "nt" else shlex.join(evaluation_cmd))
    if not args.execute:
        print("DRY RUN: no imports of training libraries, output writes, downloads, GPU use, or test-image access. Add --execute to proceed.")
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.path.insert(0, str(ROOT))
    targets = [run_path(args, seed, stage) for stage in stages for seed in args.seeds]
    if args.stage in ("evaluate", "all"):
        targets.append(args.runs_root / "evaluation")
    for target in targets:
        require(not target.exists(), f"Refusing existing output: {target}. Use a fresh --runs-root; automatic resume is intentionally unsupported.")
    manifests = validate_training_identities(args)
    for stage in stages:
        for seed in args.seeds:
            cmd = command(args, seed, stage)
            subprocess.run(cmd, cwd=ROOT, check=True)
            folder = run_path(args, seed, stage)
            if stage == "vit":
                same_folds(read_json(folder / "folds.json"), manifests[seed])
            write_json(folder / "REVISION_WRAPPER.json", dict(seed=seed, stage=stage,
                       fold_sha256=sha256(fold_path(seed)), source_sha256=source_hashes(), device=args.device))
    if args.stage in ("evaluate", "all"):
        evaluate(args, manifests)


if __name__ == "__main__":
    main()

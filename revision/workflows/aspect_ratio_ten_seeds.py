"""Extend the locked letterbox pilot to the same ten square-resize seeds."""
import json, hashlib, subprocess
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__import__('os').environ.get('ORGANOID_PROJECT_ROOT', str(Path(__file__).resolve().parents[2])))
OUT=ROOT/'runs/reviewer_aspect_ratio_ten_seeds_20261031'
SEEDS=[70384,12731,76313,98831,80324,104729,130363,155921,180749,205019]
PILOT=ROOT/'runs/reviewer_aspect_ratio_seed70384_20261029'
TEMPLATE=ROOT/'aspect_ratio_pilot_20261029.py'
OLD_ROOT=ROOT/'runs/05_final_tuning/PROFESSOR_FEEDBACK/14_ce_vs_triplet_confirmatory/runs'
ADDED_ROOT=ROOT/'runs/reviewer_ten_seeds_extension_20261030/training'

def gather(seeds):
    rows=[]
    for seed in seeds:
        folder=PILOT if seed==70384 else OUT/f'seed_{seed}'
        assert (folder/'COMPLETE.flag').exists()
        entries=json.loads((folder/'TEST_RESULTS.json').read_text())
        assert len(entries)==2
        for entry in entries:
            rows.append(dict(seed=seed,**entry))
    return {'seeds':seeds,'rows':rows,'comparison_square_results':str(ROOT/'runs/reviewer_ten_seeds_extension_20261030/TEN_SEED_RESULTS.json')}

def main():
    assert (PILOT/'COMPLETE.flag').exists()
    template=TEMPLATE.read_text()
    assert hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()=='79f51a600fc0d8fabb501008c7460828faa387af9ab95a2cc7070137aa5f3867'
    # Verify all comparison checkpoints exist before starting new training.
    for seed in SEEDS:
        old=(OLD_ROOT if seed in SEEDS[:5] else ADDED_ROOT)/f'seed_{seed}'
        for cond in ['lambda_0','lambda_0p0025']:
            for fold in range(1,6):assert (old/cond/'folds'/f'fold_{fold}_best.pt').exists()
    OUT.mkdir(exist_ok=False)
    (OUT/'scripts').mkdir()
    protocol={'seeds':SEEDS,'reused_seed':70384,'new_training_seeds':SEEDS[1:],
              'preprocessing':'224x168 aspect-preserving bilinear resize; centered black padding to 224x224; augmentation remains before resize; no channel normalization',
              'training':'Historical settings unchanged; five folds, 20 epochs; paired CE and Triplet',
              'policy':'Run all nine additional seeds irrespective of interim results. Five-seed checkpoint is descriptive, not a stopping decision. No test-driven parameter changes.',
              'started_utc':datetime.now(timezone.utc).isoformat(),'template_sha256':hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()}
    (OUT/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
    for seed in SEEDS[1:]:
        src=template.replace('70384',str(seed))
        old_line=f"OUT=ROOT/'runs/reviewer_aspect_ratio_seed{seed}_20261029'"
        assert old_line in src
        src=src.replace(old_line,f"OUT=Path({str(OUT/f'seed_{seed}')!r})")
        old_ref=f"OLD=ROOT/'runs/05_final_tuning/PROFESSOR_FEEDBACK/14_ce_vs_triplet_confirmatory/runs/seed_{seed}'"
        assert old_ref in src
        prior=(OLD_ROOT if seed in SEEDS[:5] else ADDED_ROOT)/f'seed_{seed}'
        src=src.replace(old_ref,f"OLD=Path({str(prior)!r})")
        path=OUT/'scripts'/f'seed_{seed}.py'
        compile(src,str(path),'exec')
        path.write_text(src)
        print('START_SEED',seed,flush=True)
        subprocess.run([str(ROOT/'.venv/bin/python'),'-u',str(path)],cwd=ROOT,check=True)
        print('FINISHED_SEED',seed,flush=True)
        if seed==SEEDS[4]:
            (OUT/'FIVE_SEED_RESULTS.json').write_text(json.dumps(gather(SEEDS[:5]),indent=2))
            print('FIVE_SEEDS_COMPLETE_CONTINUING',flush=True)
    (OUT/'TEN_SEED_RESULTS.json').write_text(json.dumps(gather(SEEDS),indent=2))
    (OUT/'COMPLETE.flag').write_text(datetime.now(timezone.utc).isoformat())
    print('ALL_TEN_SEEDS_COMPLETE',flush=True)

if __name__=='__main__':main()

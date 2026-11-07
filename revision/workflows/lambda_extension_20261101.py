"""Training-only missing lambda confirmations; never modify existing experiments."""
import importlib.util
import json
import os
import subprocess
import time
import hashlib
from pathlib import Path

ROOT = Path(__import__('os').environ.get('ORGANOID_PROJECT_ROOT', str(Path(__file__).resolve().parents[2])))
OUT = ROOT / 'runs/reviewer_lambda_extension_20261101'
SPEC = importlib.util.spec_from_file_location('historical_followup', ROOT / 'lambda_followup_reference_20261101.py')
m = importlib.util.module_from_spec(SPEC)
source = SPEC.origin
code = Path(source).read_text().replace('PROJECT_ROOT = Path(__file__).resolve().parents[4]', "PROJECT_ROOT = Path(__import__('os').environ.get('ORGANOID_PROJECT_ROOT', str(Path(__file__).resolve().parents[2])))")
exec(compile(code, source, 'exec'), m.__dict__)
m.PROJECT_ROOT = ROOT
m.TRAIN_SCRIPT = ROOT / '05_final_tuning/02_ordinal_aware_triplet/train_pilot.py'
m.PYTHON = ROOT / '.venv/bin/python'
m.OUTPUT_ROOT = OUT
m.STATUS_PATH = OUT / 'STATUS.json'
m.LOCK_PATH = OUT / 'runner.lock'
m.BASELINE_MULTISEED_ROOT = ROOT / 'runs/05_final_tuning/PROFESSOR_FEEDBACK/04_triplet_final_tuning/02_lr_wd_multiseed_confirmation/runs'

def wait_capacity(action):
    good = 0
    while good < 3:
        result = subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'], text=True)
        free, util = [int(x.strip()) for x in result.splitlines()[0].split(',')]
        processes = m.gpu_processes()
        ready = free >= 12000 and util <= 85 and len(processes) <= 2
        good = good + 1 if ready else 0
        m.update_status('checking_capacity' if ready else 'waiting_for_capacity', action=action, free_mib=free, utilization=util, compute_processes=processes)
        time.sleep(5 if ready else 60)

m.wait_for_idle_gpu = wait_capacity

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m.acquire_lock()
    try:
        digest = hashlib.sha256(m.TRAIN_SCRIPT.read_bytes()).hexdigest()
        assert digest == '440224281d06e42f4138b81068e2013bddac11eaf5f2f6db35127d1de8629287', digest
        for seed in m.SEEDS:
            assert (m.BASELINE_MULTISEED_ROOT / f'seed_{seed}/lr_3e-5__wd_0p01/lambda_0/oof_metrics.json').is_file()
        m.write_json(OUT / 'PLAN.json', dict(lambdas=[0.0005,0.001,0.01], seeds=list(m.SEEDS), folds=3, epochs=20, margin=0.05, trainer_sha256=digest, test_status='NOT_ACCESSED', existing_runs='read-only; CE baselines reused', capacity_rule='before each job: >=12000 MiB free, <=85% utilization, <=2 other compute processes for three samples'))
        for value in [0.0005,0.001,0.01]:
            m.run_multiseed(value)
        summaries = [m.read_json(OUT / 'multiseed' / m.lambda_key(v) / 'summary/LAMBDA_MULTISEED_SELECTION.json') for v in [0.0005,0.001,0.01]]
        m.write_json(OUT / 'RESULTS.json', {'additional_results':summaries,'test_status':'NOT_ACCESSED'})
        m.update_status('complete', completed_combinations=9)
        (OUT / 'COMPLETE.flag').write_text('9 combinations, 27 folds complete; test set not accessed\n')
    except Exception as exc:
        m.update_status('failed', error=repr(exc))
        raise
    finally:
        m.LOCK_PATH.unlink(missing_ok=True)

if __name__ == '__main__':
    main()

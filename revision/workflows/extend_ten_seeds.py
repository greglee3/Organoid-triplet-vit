"""Prespecified five additional paired seeds; no hyperparameter search."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json, shlex, subprocess, importlib.util, hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy import stats

ROOT=Path(__import__('os').environ.get('ORGANOID_PROJECT_ROOT', str(Path(__file__).resolve().parents[2])))
OUT=ROOT/'runs/reviewer_ten_seeds_extension_20261030'
OLD_SEEDS=[70384,12731,76313,98831,80324]
NEW_SEEDS=[104729,130363,155921,180749,205019]

def main():
    OUT.mkdir(exist_ok=False)
    script=ROOT/'05_final_tuning/PROFESSOR_FEEDBACK/14_ce_vs_triplet_confirmatory/run_confirmation.sh'
    source=script.read_text()
    # Use the historical command verbatim; substitute only seed and output.
    command=source.split('  "${PYTHON}" -u "${TRAINER}"',1)[1].split('\ndone',1)[0]
    command='"${PYTHON}" -u "${TRAINER}"'+command
    command=command.replace('\\\n',' ')
    manifest={'old_seeds':OLD_SEEDS,'additional_seeds':NEW_SEEDS,'total_seeds':10,
              'policy':'All five additional seeds selected before execution; run all regardless of results. Same original square resize and training recipe. No test-based selection.',
              'started_utc':datetime.now(timezone.utc).isoformat(),
              'source_command_sha256':hashlib.sha256(script.read_bytes()).hexdigest()}
    (OUT/'PROTOCOL.json').write_text(json.dumps(manifest,indent=2))
    for seed in NEW_SEEDS:
        replacements={'PYTHON':ROOT/'.venv/bin/python','TRAINER':ROOT/'05_final_tuning/02_ordinal_aware_triplet/train_pilot.py',
                      'PROJECT_ROOT':ROOT,'SEED_ROOT':OUT/'training'/f'seed_{seed}','SEED':seed}
        cmd=command
        for k,v in replacements.items():cmd=cmd.replace('${'+k+'}',str(v))
        assert '${' not in cmd
        print('START_ADDITIONAL_SEED',seed,flush=True)
        subprocess.run(shlex.split(cmd),cwd=ROOT,check=True)
        print('TRAINING_COMPLETE',seed,flush=True)
    spec=importlib.util.spec_from_file_location('extension_evaluator',ROOT/'reviewer1_multiseed_test_20261029.py')
    ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    ev.SEEDS=NEW_SEEDS;ev.RUNS=OUT/'training';ev.OUT=OUT/'test'
    ev.main()
    old=json.loads((ROOT/'runs/reviewer1_multiseed_test_20261029/RESULTS.json').read_text())['rows']
    new=json.loads((OUT/'test/RESULTS.json').read_text())['rows']
    rows=old+new
    assert len(rows)==20
    lookup={(r['seed'],r['model']):r['metrics'] for r in rows}
    assert len(lookup)==20
    summary={}
    for key in ['accuracy','macro_f1','c_recall','c_precision','c_specificity','c_f1']:
        ce=np.array([lookup[s,'CE-only ViT'][key] for s in OLD_SEEDS+NEW_SEEDS])*100
        tr=np.array([lookup[s,'Triplet-ViT'][key] for s in OLD_SEEDS+NEW_SEEDS])*100
        diff=tr-ce;tt=stats.ttest_rel(tr,ce)
        ci=stats.t.interval(.95,9,loc=diff.mean(),scale=stats.sem(diff))
        summary[key]={'ce_mean':float(ce.mean()),'ce_sd':float(ce.std(ddof=1)),
                      'triplet_mean':float(tr.mean()),'triplet_sd':float(tr.std(ddof=1)),
                      'difference_pp':float(diff.mean()),'ci95_pp':list(map(float,ci)),
                      't':float(tt.statistic),'df':9,'p':float(tt.pvalue)}
    ordered=sorted(summary,key=lambda k:summary[k]['p']);running=0.
    for i,k in enumerate(ordered):
        running=max(running,(6-i)*summary[k]['p']);summary[k]['p_holm']=min(1.,running)
    (OUT/'TEN_SEED_RESULTS.json').write_text(json.dumps({'rows':rows,'summary':summary,'note':'Post-hoc extension after inspection of original five seeds. CIs concern seed variation on fixed test images, not test sampling.'},indent=2))
    (OUT/'COMPLETE.flag').write_text(datetime.now(timezone.utc).isoformat())
    print('ALL_COMPLETE',flush=True)

if __name__=='__main__':main()

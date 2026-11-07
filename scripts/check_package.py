"""Static, dependency-free checks. Does not train or access data."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    files = [p for p in ROOT.rglob('*') if p.is_file()
        and p != ROOT / 'data/Asano_Full_Data.zip'
        and not any(part in {'.git', '.venv', '__pycache__', 'generated', 'runs'}
                    for part in p.relative_to(ROOT).parts)]
    for p in files:
        if p.suffix == '.py':
            ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))
        if p.suffix == '.json':
            json.loads(p.read_text(encoding='utf-8-sig'))
        assert p.suffix.lower() not in {'.pt', '.pth', '.ckpt', '.zip', '.docx', '.pptm', '.pem', '.key'}
        assert p.stat().st_size < 5_000_000, f'Unexpected large file: {p.name}'
    manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text())
    for entry in manifest['files']:
        path = ROOT / entry['path']
        assert path.is_file(), entry['path']
        content = path.read_bytes()
        if 'packaged_lf_sha256' in entry:
            actual = hashlib.sha256(content.replace(b'\r\n', b'\n')).hexdigest()
            expected = entry['packaged_lf_sha256']
        else:
            actual = hashlib.sha256(content).hexdigest()
            expected = entry.get('packaged_sha256', entry['source_sha256'])
        assert actual == expected, entry['path']
    folds = json.loads((ROOT / 'reference_results/seed42_folds.json').read_text())
    assert folds['seed'] == 42
    assert len(folds['ordered_samples']) == 1200
    revision = ROOT / 'revision/SOURCE_MANIFEST.json'
    if revision.exists():
        for entry in json.loads(revision.read_text())['files']:
            p = ROOT / entry['path']
            assert p.is_file(), entry['path']
            actual = hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            assert actual == entry['packaged_lf_sha256'], entry['path']
        ordered = None
        for seed in [70384,12731,76313,98831,80324,104729,130363,155921,180749,205019]:
            obj=json.loads((ROOT/f'revision/folds/seed_{seed}.json').read_text())
            assert obj['seed']==seed and len(obj['folds'])==5
            if ordered is None:ordered=obj['ordered_samples']
            assert ordered==obj['ordered_samples'] and len(ordered)==1200
            validation=[]
            for fold in obj['folds']:
                train=set(fold['train_indices']);val=set(fold['validation_indices'])
                assert len(train)==960 and len(val)==240 and not train&val
                assert train|val==set(range(1200))
                validation.extend(val)
            assert sorted(validation)==list(range(1200))
    print(f'PASS: syntax, JSON, source manifest, 1,200-sample fold inventory; {len(files)} files checked.')

if __name__ == '__main__':
    main()

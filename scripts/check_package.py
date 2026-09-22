"""Static, dependency-free checks. Does not train or access data."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    files = [p for p in ROOT.rglob('*') if p.is_file() and not any(
        part in {'.git', '.venv', '__pycache__', 'generated', 'runs'} for part in p.relative_to(ROOT).parts)]
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
    print(f'PASS: syntax, JSON, source manifest, 1,200-sample fold inventory; {len(files)} files checked.')

if __name__ == '__main__':
    main()

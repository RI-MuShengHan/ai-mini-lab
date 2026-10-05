"""Run: python -m mini_lab.evaluate [digits|wine|search|all]."""
import argparse
import json
from pathlib import Path
import platform
import sklearn
from mini_lab import digits, wine
from mini_lab.search import NoteIndex

ROOT = Path(__file__).resolve().parents[1]


def search_report():
    index = NoteIndex(ROOT / 'data/demo_notes')
    cases = json.loads((ROOT / 'data/search_eval.json').read_text(encoding='utf-8'))
    rows = []
    for case in cases:
        hits = index.search(case['query'])
        matches = [i+1 for i, hit in enumerate(hits)
                   if (hit['source'], hit['heading']) == (case['source'], case['heading'])]
        rows.append({**case, 'rank': matches[0] if matches else None})
    return {'dataset': '8 authored demo-note queries; not an external benchmark',
            'count': len(rows), 'hit_at_1': sum(x['rank']==1 for x in rows)/len(rows),
            'hit_at_3': sum(x['rank'] is not None for x in rows)/len(rows), 'queries': rows,
            'note': 'Lexical TF-IDF retrieval, no LLM generation or semantic embeddings.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', choices=['digits','wine','search','all'], default='all', nargs='?')
    args = parser.parse_args()
    out = ROOT / 'artifacts'; out.mkdir(exist_ok=True)
    for name, fn in [('digits', lambda: digits.train()[-1]),
                     ('wine', lambda: wine.train()[-1]), ('search', search_report)]:
        if args.project not in ('all', name): continue
        report = fn()
        report['environment'] = {'python': platform.python_version(), 'sklearn': sklearn.__version__}
        (out / f'{name}_metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(name, json.dumps({k:v for k,v in report.items() if k not in ('confusion_matrix','queries')},ensure_ascii=False), flush=True)


if __name__ == '__main__': main()

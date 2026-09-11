"""Recompute all frozen final success rates from per-case records (stdlib only)."""
from pathlib import Path
import csv,json,math,statistics

def main():
    root=Path(__file__).resolve().parents[1]
    exp=root/'experiments/strict_navigation/final_confirmation_v1'
    q=json.loads((exp/'protocol.json').read_text());final=json.loads((exp/'results.json').read_text())
    compact=json.loads((root/'results/final_cases.json').read_text())
    assert compact['schema_version']==1
    columns=['episode','start_step','success','initial_success','across_wall']
    assert compact['columns']==columns
    assert len(compact['batches'])==len(q['evaluations'])==238
    assert len({b['tag'] for b in compact['batches']})==238
    groups={}
    for j,b in zip(q['evaluations'],compact['batches']):
        assert all(b[k]==j[k] for k in ['tag','domain','training_seed','method'])
        assert len(b['rows'])==50 and all(len(row)==len(columns) for row in b['rows'])
        rows=[dict(zip(columns,row)) for row in b['rows']]
        assert all(r[k] in (True,False) for r in rows for k in columns[2:])
        key=(j['domain'],str(j['training_seed']),j['method']);groups.setdefault(key,[]).extend(rows)
    sr=lambda rows:100*sum(bool(x['success']) for x in rows)/len(rows) if rows else None
    for (domain,seed,method),rows in groups.items():
        expected=final['results'][domain][seed][method]
        assert len(rows)==expected['n']
        assert len({(x['episode'],x['start_step']) for x in rows})==len(rows)
        for field,rr in [('sr',rows),('cross_wall_sr',[x for x in rows if x['across_wall']]),
                         ('same_side_sr',[x for x in rows if not x['across_wall']]),
                         ('sr_excluding_initial',[x for x in rows if not x['initial_success']])]:
            value=sr(rr);recorded=expected[field]
            assert value is None and recorded is None or math.isclose(value,recorded,abs_tol=1e-9),(domain,seed,method,field)
    csvrows=list(csv.DictReader((root/'results/final_summary.csv').open()))
    assert len(csvrows)==25
    for row in csvrows:
        values=[sr(groups[(row['domain'],s,row['method'])]) for s in ['3072','3073','3074']]
        assert math.isclose(statistics.mean(values),float(row['mean_sr_percent']),abs_tol=1e-9)
        assert math.isclose(statistics.stdev(values),float(row['sample_sd_percent']),abs_tol=1e-9)
    assert sum(len(x) for x in groups.values())==11900 and len(groups)==85
    print(json.dumps({'passed':True,'batches':238,'episode_evaluations':sum(len(x) for x in groups.values()),'groups':len(groups),'summary_rows':len(csvrows),'note':'Recomputed from compact case records, not a raw trace re-audit. Cases reused across seeds/methods/maps; not independent sample count.'},indent=2))

if __name__=='__main__':main()

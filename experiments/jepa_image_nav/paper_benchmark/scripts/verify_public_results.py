"""Verify public hashes and scalar scores; does not rerun private simulator traces."""
import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read(p):
    return json.loads(p.read_text())


def hashfile(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x]


def compare(data, summary):
    assert len(data)==summary['valid_episodes']
    assert len({r['episode_id'] for r in data})==len(data)
    groups=defaultdict(list)
    for r in data:groups[r['scene_id']].append(r)
    assert len(groups)==43
    assert sum(r['success'] for r in data)==summary['successes']
    for field, key in [('success','SR'),('spl','SPL')]:
        ep=sum(r[field] for r in data)/len(data)
        sc=sum(sum(r[field] for r in g)/len(g) for g in groups.values())/len(groups)
        assert math.isclose(ep,summary['episode_weighted_'+key],abs_tol=1e-12)
        assert math.isclose(sc,summary['scene_weighted_'+key],abs_tol=1e-12)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-hashes',action='store_true')
    opt=parser.parse_args()
    metrics=rows(ROOT/'results/episode_metrics.jsonl')
    assert len(metrics)==1719
    stats=read(ROOT/'results/compact_statistics.json')
    models={m['model']:m for m in stats['models']}
    assert stats['bootstrap_replicates']==10000 and stats['bootstrap_seed']==20261017
    assert {r['model'] for r in metrics}==set(models)
    for name,m in models.items():compare([r for r in metrics if r['model']==name],m)
    ex=read(ROOT/'results/implementation_exceptions.json')
    assert len(ex)==1 and ex[0]['score'] is None and not ex[0]['counted_as_navigation_failure']
    index=read(ROOT/'manifests/selected_case_ids.json')
    for task,panel in index.items():
        assert len(panel)==430 and len({r['episode_id'] for r in panel})==430
        assert len(Counter(r['scene_id'] for r in panel))==43
        assert set(Counter(r['scene_id'] for r in panel).values())=={10}
    for pair in stats['paired_comparisons']:
        first,second=('fgprompt','debit') if pair['task']=='place' else ('goat','unigoal')
        a={r['episode_id']:r for r in metrics if r['model']==first}
        b={r['episode_id']:r for r in metrics if r['model']==second}
        common=set(a)&set(b)
        assert len(common)==pair['common_valid_episodes']
        digest=hashlib.sha256(('\n'.join(sorted(common))+'\n').encode()).hexdigest()
        assert digest==pair['common_valid_ids_sha256']
        compare([a[k] for k in sorted(common)],pair['first_common_valid_statistics'])
        compare([b[k] for k in sorted(common)],pair['second_common_valid_statistics'])
    off=rows(ROOT/'results/official_gibson_episode_metrics.jsonl')
    gate=read(ROOT/'results/official_gibson_fgprompt.json')
    assert len(off)==4901 and len({(r['scene_id'],r['episode_id']) for r in off})==4901
    assert len({r['scene_id'] for r in off})==14
    assert sum(r['success'] for r in off)==gate['successes']==4083
    assert math.isclose(sum(r['success'] for r in off)/4901,gate['episode_weighted_SR'],abs_tol=1e-12)
    assert math.isclose(sum(r['spl'] for r in off)/4901,gate['episode_weighted_SPL'],abs_tol=1e-12)
    groups=defaultdict(list)
    for r in off:groups[r['scene_id']].append(r)
    for f,k in [('success','SR'),('spl','SPL')]:
        actual=sum(sum(r[f] for r in g)/len(g) for g in groups.values())/14
        assert math.isclose(actual,gate['scene_weighted_'+k],abs_tol=1e-12)
    accepted=read(ROOT/'results/acceptance.json')
    assert accepted['exception_aware_assessment_passed'] and all(c['passed'] for c in accepted['checks'])
    assert accepted['valid_completed']==1719 and not accepted['original_strict_1720_valid_gate_passed']
    for d in read(ROOT/'results/diagnostics.json').values():
        assert sum(d['termination_partition'].values())==d['valid_episodes']
    forbidden_keys={'before','after','rotation_xyzw_before','rotation_xyzw_after','goal_position',
                    'start_position','camera_position_after','valid_approach_views','prompt','messages',
                    'api_key','authorization','image_base64','response_text'}
    def check_keys(data):
        if isinstance(data,dict):
            assert not (set(data)&forbidden_keys), 'Private payload field in public JSON'
            for v in data.values():check_keys(v)
        elif isinstance(data,list):
            for v in data:check_keys(v)
    for p in ROOT.rglob('*'):
        if not p.is_file():continue
        assert p.suffix in {'.py','.md','.json','.jsonl'}, 'Unexpected artifact extension'
        assert '.private.' not in p.name and '.steps.' not in p.name
        text=p.read_text()
        assert not re.search(r'sk-or-v1-[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{24,}|-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----',text)
        if p.suffix=='.json':check_keys(json.loads(text))
        if p.suffix=='.jsonl':
            for row in rows(p):check_keys(row)
        if p.suffix=='.md':
            for link in re.findall(r'\]\(([^)]+)\)',text):
                if re.match(r'https?://|#',link):continue
                assert (p.parent/link.split('#')[0]).exists(), 'Broken relative link: '+link
    checksum=ROOT/'manifests/public_files.sha256.json'
    actual={str(p.relative_to(ROOT)):hashfile(p) for p in sorted(ROOT.rglob('*'))
            if p.is_file() and p!=checksum}
    if opt.freeze_hashes:
        payload=json.dumps(actual,indent=2,sort_keys=True)+'\n'
        if checksum.exists():assert checksum.read_text()==payload
        else:checksum.write_text(payload)
    assert read(checksum)==actual,'Public file digest mismatch'
    print(json.dumps(dict(passed=True,valid_hssd_scores=1719,original_gibson_scores=4901,
        unscored_native_exceptions=1,files_checked=len(actual),secrets_and_private_payload_scan_passed=True,
        scalar_SR_SPL_recomputed=True,bootstrap_intervals_not_recomputed=True,
        simulator_and_policy_not_rerun=True)))


if __name__=='__main__':main()

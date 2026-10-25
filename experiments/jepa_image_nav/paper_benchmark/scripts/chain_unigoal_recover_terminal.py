"""Validate narrowly scoped completed episode before recovering a post-terminal error."""
import ast,hashlib,json,re
from pathlib import Path
from chain_archive_attempt import archive
ROOT=Path('/root/autodl-tmp/jepa_instance_bench_20261016')
episode='hssd-instance-102344022-77-1-1'
folder=ROOT/'results/unigoal_native_instance_hssd300/episodes'/episode
assert json.loads((ROOT/'results/unigoal_flow_gate.json').read_text())['passed']
failure=folder/'runtime_failure.json';raw=failure.read_bytes();value=json.loads(raw)
assert value['error']=='TypeError' and value['detail']=='list indices must be integers or slices, not str'
records=json.loads((folder/'records.json').read_text());assert len(records)==1
record=records[0]
assert record['episode_id']==episode and record['status']=='complete' and record['steps']==500
trace=folder/(episode+'.steps.jsonl');trace_data=trace.read_bytes();steps=[json.loads(x) for x in trace_data.decode().splitlines()]
assert len(steps)==500 and steps[-1]['t']==499 and record['forced_budget_termination']
source=(ROOT/'repos/UniGoal/main.py').read_text();tree=ast.parse(source)
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
loop=next(n for n in main.body if isinstance(n,ast.While))
done=next(n for n in loop.body if isinstance(n,ast.If) and ast.unparse(n.test)=='done')
terminal=next(n for n in done.body if isinstance(n,ast.If) and ast.unparse(n.test)=='len(episode_success) == args.num_episodes')
assert ast.unparse(terminal.body[0])=='finished = True'
calls=[n for n in ast.walk(done) if isinstance(n,ast.Call) and ast.unparse(n.func)=='graph.set_image_goal']
assert len(calls)==1 and calls[0].lineno>terminal.lineno
log=(folder/'native_run.log').read_text()
assert re.search(r'UniGoal/main\.py", line '+str(calls[0].lineno)+r', in main',log)
assert "graph.set_image_goal(infos['instance_imagegoal'])" in log and 'goalgraphdecomposer.py' in log
audit={'recovery':'Post-terminal unused graph initialization only; full completed trajectory still requires normal independent acceptance',
    'episode_id':episode,'completed_steps':500,'original_runtime_failure_sha256':hashlib.sha256(raw).hexdigest(),
    'original_records_sha256':hashlib.sha256((folder/'records.json').read_bytes()).hexdigest(),
    'original_trace_sha256':hashlib.sha256(trace_data).hexdigest(),
    'native_main_sha256':hashlib.sha256(source.encode()).hexdigest(),'post_terminal_error_line':calls[0].lineno,
    'scores_or_trace_rewritten':False,'native_model_decisions_replayed':False}
archive(folder,['runtime_failure.json'])
(folder/'post_terminal_recovery.json').write_text(json.dumps(audit,indent=2))
(ROOT/'results/unigoal_post_terminal_recovery.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))

"""Offline regression for terminal-only break and original bounded parser retry."""
import ast,copy,importlib.util,json,tempfile
from pathlib import Path
from chain_unigoal_flow import terminal_tree,install_schema
ROOT=Path('/root/autodl-tmp/jepa_instance_bench_20261016');REPO=ROOT/'repos/UniGoal'
source=(REPO/'main.py').read_text();original=ast.parse(source)
native=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='main')
modified=terminal_tree(source).body[0]
terminal=next(n for n in ast.walk(modified) if isinstance(n,ast.If) and ast.unparse(n.test)=='len(episode_success) == args.num_episodes')
assert isinstance(terminal.body.pop(),ast.Break)
assert ast.dump(native,include_attributes=False)==ast.dump(modified,include_attributes=False)
path=REPO/'src/graph/goalgraphdecomposer.py'
spec=importlib.util.spec_from_file_location('native_goal_decomposer_gate',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
goal={'nodes':[{'id':'chair'}],'edges':[]};valid={'subgraph_1':copy.deepcopy(goal)}
expected=module.GoalGraphDecomposer(lambda prompt:json.dumps(valid)).goal_decomposition(copy.deepcopy(goal))
with tempfile.TemporaryDirectory(prefix='unigoal_flow_test_') as directory:
    install_schema(module,path,Path(directory))
    actual=module.GoalGraphDecomposer(lambda prompt:json.dumps(valid)).goal_decomposition(copy.deepcopy(goal))
    assert actual==expected
    replies=iter(['[]','{"nodes":[]}','{"x":{"nodes":[1]}}',json.dumps(valid)])
    calls=[]
    def llm(prompt):calls.append(prompt);return next(replies)
    assert module.GoalGraphDecomposer(llm).goal_decomposition(copy.deepcopy(goal))==expected
    assert len(calls)==4 and len(set(calls))==1
    count=[]
    fallback=module.GoalGraphDecomposer(lambda prompt:count.append(prompt) or '[]').goal_decomposition(copy.deepcopy(goal))
    assert len(count)==10 and fallback=={'subgraph_1':goal}
result={'passed':True,'paid_requests':0,'model_calls':0,'AST_only_change_is_final_break':True,
    'valid_native_graph_response_unchanged':True,'bad_shapes_retried_with_identical_prompt':True,
    'original10attempt_bound_and_fallback_retained':True}
(ROOT/'results/unigoal_flow_gate.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

"""Confirm the repair changes collision telemetry only, and preserves frozen inputs/results."""
import ast
import json
from pathlib import Path
from paper_freeze import ROOT, sha, frozen


def main():
    old = ROOT/'operational_snapshots/pre_repair_20261019/paper_sim.py'
    new = ROOT/'tools/paper_sim.py'
    a = ast.parse(old.read_text()); b = ast.parse(new.read_text())
    class ReplaceAttribute(ast.NodeTransformer):
        count = 0
        def visit_Attribute(self,node):
            if isinstance(node.value,ast.Name) and node.value.id == 'sim' and node.attr == 'previous_step_collided':
                self.count += 1
                return ast.parse("obs['collided']",mode='eval').body
            return self.generic_visit(node)
    class RemoveNewAssertion(ast.NodeTransformer):
        count = 0
        def visit_If(self,node):
            expected = ast.parse("if action == 1:\n    assert isinstance(obs['collided'], (bool, np.bool_))").body[0]
            if ast.dump(node,include_attributes=False) == ast.dump(expected,include_attributes=False):
                self.count += 1
                return None
            return self.generic_visit(node)
    replace = ReplaceAttribute(); strip = RemoveNewAssertion()
    a = replace.visit(a); b = strip.visit(b)
    assert replace.count == strip.count == 1
    assert ast.dump(a,include_attributes=False) == ast.dump(b,include_attributes=False)
    collision = json.loads((ROOT/'results/repair_20261019/collision_interface_gate.json').read_text())
    assert collision['passed'] and collision['sim_sha256'] == sha(new)
    inventory = json.loads((ROOT/'results/repair_20261019/official_trace_inventory.private.json').read_text())
    assert all(sha(Path(p)) == h for p,h in inventory['preserved_file_hashes'].items())
    legacy = json.loads((ROOT/'results/compact430_legacy_pause_requested.json').read_text())
    assert all(sha(Path(p)) == h for p,h in legacy['protected_completed_files'].items())
    result = dict(passed=True,AST_difference_only_native_collision_read_and_type_check=True,
        original_sim_sha256=sha(old),repaired_sim_sha256=sha(new),
        official_completed_files_byte_preserved=len(inventory['preserved_file_hashes']),
        legacy_completed_files_byte_preserved=len(legacy['protected_completed_files']),
        frozen_contract_sha256=sha(ROOT/'protocol/common_v1.json'),
        compact_manifest_sha256=sha(ROOT/'manifests/hssd_val43_compact430_freeze.json'),
        models_actions_sensors_scoring_and_episode_budget_unchanged=True)
    frozen(ROOT/'results/repair_20261019/repair_integrity_gate.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()

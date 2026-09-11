"""Audit real execution before first terminal; allow official post-terminal NaN padding."""
import numpy as np
def audit_executed_trace(trace,rows):
    assert {'actions','terminated','proprio'}.issubset(trace.files)
    actions=trace['actions'];term=trace['terminated'];pos=trace['proprio']
    assert actions.shape[:2]==term.shape[:2]==pos.shape[:2]
    assert term.ndim==2 and term.shape[1]==len(rows)
    assert np.isfinite(term).all() and np.isfinite(pos).all()
    assert not np.isinf(actions).any()
    padding=0;executed=0
    for i,row in enumerate(rows):
        done=np.flatnonzero(term[:,i])
        n=int(done[0]+1) if len(done) else len(term)
        assert row['steps']==n and row['success']==bool(len(done)),i
        assert np.isfinite(actions[:n,i]).all(),('nonfinite active action',i,n)
        padding+=int(np.isnan(actions[n:,i]).sum())
        executed+=int(actions[:n,i].size)
    return {'passed':True,'executed_action_values':executed,'post_terminal_nan_padding_values':padding,'active_nonfinite_values':0,'rule':'All states finite; actions finite through first terminal inclusive. Only post-terminal NaN padding allowed; infinity never allowed.'}

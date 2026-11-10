"""Closed-form targeted head edit, using only synthetic numbers in this demo.

No optimizer, model or external service. This constructs a local parameter patch,
not a learned semantic rule. Construction controls are not held-out evidence.
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wsp.weight_edit import dot, edit_output_head, logits


def direction_for_target(target, controls, margin_delta, max_update_norm=10.0, tolerance=1e-10):
    """Find v with v·target=delta, v·controls=0; reject degenerate/large edits.

    Modified Gram-Schmidt produces a basis for the control span. Reorthogonalize
    twice to reduce roundoff. Production adapters need numerical conditioning tests.
    """
    if not target or any(len(x) != len(target) for x in controls):
        raise ValueError('Inconsistent dimensions')
    if not all(math.isfinite(x) for row in [target,*controls] for x in row):
        raise ValueError('Finite vectors required')
    if not math.isfinite(margin_delta) or not math.isfinite(max_update_norm) or max_update_norm <= 0:
        raise ValueError('Invalid margin or edit-norm limit')
    basis=[]
    for control in controls:
        q=list(control)
        for _ in range(2):
            for b in basis:
                coefficient=dot(q,b)
                q=[a-coefficient*c for a,c in zip(q,b)]
        norm=math.sqrt(dot(q,q))
        if norm > tolerance:
            basis.append([x/norm for x in q])
    u=list(target)
    for _ in range(2):
        for b in basis:
            coefficient=dot(u,b)
            u=[a-coefficient*c for a,c in zip(u,b)]
    norm2=dot(u,u)
    if norm2 <= tolerance**2 * max(1.0,dot(target,target)):
        raise ValueError('Target lies within or too close to the control span; do not force an edit')
    v=[margin_delta*x/norm2 for x in u]
    update_norm=math.sqrt(dot(v,v)/2)
    if update_norm > max_update_norm:
        raise ValueError('Required output-head update exceeds the prespecified norm cap')
    return v


def demo():
    target=[1.,2.,3.]
    controls=[[1.,0.,0.],[0.,1.,0.]]
    v=direction_for_target(target,controls,margin_delta=1.5)
    base=[[.1,.2,.3],[-.2,.1,.4],[.5,.1,-.2]]
    changed=edit_output_head(base,v,yes=0,no=1)
    delta=lambda h: (logits(changed,h)[0]-logits(changed,h)[1])-(logits(base,h)[0]-logits(base,h)[1])
    assert abs(delta(target)-1.5)<1e-12
    assert all(abs(delta(c))<1e-12 for c in controls)
    return {'synthetic_only':True,'target_margin_change':delta(target),
            'construction_control_changes':[delta(c) for c in controls],
            'heldout_model_behavior_tested':False,'optimizer_used':False}


if __name__=='__main__':
    import json
    print(json.dumps(demo(),indent=2))

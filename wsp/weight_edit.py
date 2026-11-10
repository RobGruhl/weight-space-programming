"""Pure-Python algebra only. No transformer, checkpoint, network or inference dependency.

This is a reference implementation for a later approved model adapter, NOT a
replication of any paper and NOT evidence of improved email judgment.
"""
import math


def dot(a, b):
    if len(a) != len(b):
        raise ValueError('Vector dimensions differ')
    return sum(x*y for x, y in zip(a, b))


def edit_output_head(weights, direction, yes, no, alpha=1.0, *, tied=False, detached_output_copy=False):
    """Return an edited dense copy: W + alpha/2 * (e_yes-e_no) direction^T.

    A caller with tied embeddings must explicitly create/use an independent output
    head. This helper cannot check tensor storage aliases or packed quantization.
    Such checks remain mandatory in a real model adapter.
    """
    if tied and not detached_output_copy:
        raise ValueError('Tied embeddings require an explicitly detached output-head copy')
    if yes == no or not 0 <= yes < len(weights) or not 0 <= no < len(weights):
        raise ValueError('Two distinct, valid candidate token IDs required')
    if not direction or any(len(row) != len(direction) for row in weights):
        raise ValueError('Invalid head dimensions')
    if not all(math.isfinite(x) for row in weights for x in row) or not all(math.isfinite(x) for x in direction) or not math.isfinite(alpha):
        raise ValueError('Finite values required')
    result = [list(row) for row in weights]
    for j, value in enumerate(direction):
        result[yes][j] += alpha * value / 2
        result[no][j] -= alpha * value / 2
    return result


def logits(head, hidden):
    return [dot(row, hidden) for row in head]


if __name__ == '__main__':
    import json
    base = [[.1,.2,.3],[-.3,.2,.1],[.4,-.2,.5]]
    h, direction, alpha = [.2,-.7,.1], [1.0,.5,-.2], .8
    edited = edit_output_head(base, direction, yes=0, no=1, alpha=alpha)
    before, after = logits(base,h), logits(edited,h)
    expected = alpha * dot(direction,h)
    observed = (after[0]-after[1]) - (before[0]-before[1])
    assert math.isclose(observed, expected, abs_tol=1e-12)
    print(json.dumps({'synthetic_only': True, 'expected_incremental_contrast':expected,
                      'observed_incremental_contrast':observed, 'unchanged_other_logit':after[2]==before[2],
                      'model_loaded': False, 'email_inference_run': False}, indent=2))

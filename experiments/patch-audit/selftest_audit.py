"""Synthetic self-test for audit_p50c_stdlib.py: builds a tiny fake safetensors + sparse edit, expects PASS, then corrupts the manifest and expects FAIL."""
import hashlib, json, struct, subprocess, sys, tempfile, zipfile
from pathlib import Path
here = Path(__file__).resolve().parent
temporary = tempfile.TemporaryDirectory(prefix="wsp-audit-selftest-")
tmp = Path(temporary.name)
rows, cols = 4, 6
name = 'model.language_model.layers.15.mlp.down_proj.weight'
data = struct.pack('<24H', *range(1, 25))
header = {name: {'dtype': 'BF16', 'shape': [rows, cols], 'data_offsets': [0, len(data)]}}
h = json.dumps(header).encode(); h += b' ' * (-len(h) % 8)
(tmp / 'model.safetensors').write_bytes(struct.pack('<Q', len(h)) + h + data)
idx, val = [2, 7, 13], [100, 8, 200]          # base[7] == 8 already, so only two entries really change
def npy(descr, arr, code):
    hdr = ("{'descr': '%s', 'fortran_order': False, 'shape': (%d,), }" % (descr, len(arr))).encode('latin1')
    hdr += b' ' * (64 - ((10 + len(hdr) + 1) % 64)) + b'\n'
    return b'\x93NUMPY\x01\x00' + struct.pack('<H', len(hdr)) + hdr + struct.pack('<%d%s' % (len(arr), code), *arr)
edit = tmp / 'edit'; edit.mkdir()
with zipfile.ZipFile(edit / 'edit.npz', 'w') as z:
    z.writestr('t0_idx.npy', npy('<i8', idx, 'q')); z.writestr('t0_val.npy', npy('<u2', val, 'H'))
edited = bytearray(data)
for j, v in zip(idx, val): struct.pack_into('<H', edited, 2 * j, v)
manifest = {'tensors': [{'name': name, 'shape': [rows, cols], 'dtype': 'bfloat16', 'changed_entries': 2, 'changed_columns': 2, 'changed_rows': 2,
             'base_sha256': hashlib.sha256(data).hexdigest(), 'edited_sha256': hashlib.sha256(edited).hexdigest(), 'form': 'sparse'}], 'changed_entries_total': 2}
(edit / 'manifest.json').write_text(json.dumps(manifest))
def run(label):
    r = subprocess.run([sys.executable, '-I', str(here / 'audit_p50c_stdlib.py'), str(tmp / 'model.safetensors'), str(edit), str(tmp / f'report-{label}.json')], capture_output=True, text=True)
    print(f'[{label}] exit={r.returncode}\n{r.stdout.strip()}{r.stderr.strip()}')
    return r.returncode
ok = run('clean')
manifest['tensors'][0]['edited_sha256'] = '0' * 64
(edit / 'manifest.json').write_text(json.dumps(manifest))
bad = run('corrupted')
print('SELFTEST', 'PASS' if (ok == 0 and bad == 1) else 'FAIL')

temporary.cleanup()
if not (ok == 0 and bad == 1):
    raise SystemExit(1)

"""Independent, read-only audit of the p50c patch against a local Qwen3.5-0.8B safetensors file.

Pure Python standard library on purpose: no numpy, torch, safetensors or upstream code, so this shares
no implementation with either the upstream rebuild script or the tiny-replay adapter.
Reads only. Writes a JSON report to the path given as the third argument.

    python3 -I audit_p50c_stdlib.py <model.safetensors> <p50c edit dir> <report.json>
"""
import array
import ast
import hashlib
import json
import struct
import sys
import zipfile
from pathlib import Path


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safetensors_header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def tensor_bytes(path, header, offset, name):
    info = header[name]
    a, b = info['data_offsets']
    with open(path, 'rb') as f:
        f.seek(offset + a)
        data = f.read(b - a)
    if len(data) != b - a:
        raise ValueError(f'short read for {name}')
    return info, data


def read_npy(buf):
    if buf[:6] != b'\x93NUMPY':
        raise ValueError('not an .npy member')
    major = buf[6]
    if major == 1:
        hlen = struct.unpack('<H', buf[8:10])[0]
        start = 10
    else:
        hlen = struct.unpack('<I', buf[8:12])[0]
        start = 12
    header = ast.literal_eval(buf[start:start + hlen].decode('latin1'))
    if header['fortran_order']:
        raise ValueError('fortran order not handled')
    codes = {'<i8': 'q', '<u8': 'Q', '<i4': 'i', '<u4': 'I', '<i2': 'h', '<u2': 'H', '|u1': 'B', '|i1': 'b'}
    arr = array.array(codes[header['descr']])
    arr.frombytes(buf[start + hlen:])
    if sys.byteorder != 'little':
        arr.byteswap()
    return header['descr'], header['shape'], arr


def main(model_path, edit_dir, report_path):
    model_path, edit_dir, report_path = Path(model_path), Path(edit_dir), Path(report_path)
    manifest = json.loads((edit_dir / 'manifest.json').read_text())
    npz = zipfile.ZipFile(edit_dir / 'edit.npz')
    header, offset = safetensors_header(model_path)
    report = {
        'audit': 'claude stdlib-only p50c audit',
        'model_file': str(model_path),
        'model_file_bytes': model_path.stat().st_size,
        'model_file_sha256': sha256_file(model_path),
        'edit_npz_sha256': sha256_file(edit_dir / 'edit.npz'),
        'manifest_sha256': sha256_file(edit_dir / 'manifest.json'),
        'tensors': [],
    }
    all_ok = True
    for i, t in enumerate(manifest['tensors']):
        if t['form'] != 'sparse':
            raise ValueError(f"unexpected form {t['form']}; this audit handles sparse edits only")
        info, data = tensor_bytes(model_path, header, offset, t['name'])
        rows, cols = t['shape']
        _, _, idx = read_npy(npz.read(f't{i}_idx.npy'))
        _, _, val = read_npy(npz.read(f't{i}_val.npy'))
        if len(idx) != len(val):
            raise ValueError('idx/val length mismatch')
        base_sha = hashlib.sha256(data).hexdigest()
        buf = bytearray(data)
        changed = 0
        changed_cols, changed_rows = set(), set()
        for j, v in zip(idx, val):
            old = struct.unpack_from('<H', buf, 2 * j)[0]
            if old != v:
                changed += 1
                changed_cols.add(j % cols)
                changed_rows.add(j // cols)
            struct.pack_into('<H', buf, 2 * j, v)
        edited_sha = hashlib.sha256(buf).hexdigest()
        entry = {
            'name': t['name'],
            'dtype_in_file': info['dtype'],
            'shape_in_file': info['shape'],
            'shape_match': info['shape'] == t['shape'] and info['dtype'] == 'BF16',
            'base_sha256_computed': base_sha,
            'base_sha256_manifest': t['base_sha256'],
            'base_match': base_sha == t['base_sha256'],
            'edited_sha256_computed': edited_sha,
            'edited_sha256_manifest': t['edited_sha256'],
            'edited_match': edited_sha == t['edited_sha256'],
            'touched_entries': len(idx),
            'changed_entries_computed': changed,
            'changed_entries_manifest': t['changed_entries'],
            'changed_entries_match': changed == t['changed_entries'],
            'changed_columns_computed': len(changed_cols),
            'changed_columns_manifest': t['changed_columns'],
            'changed_rows_computed': len(changed_rows),
            'changed_rows_manifest': t['changed_rows'],
        }
        all_ok &= entry['shape_match'] and entry['base_match'] and entry['edited_match'] and entry['changed_entries_match']
        report['tensors'].append(entry)
    report['changed_entries_total_computed'] = sum(e['changed_entries_computed'] for e in report['tensors'])
    report['changed_entries_total_manifest'] = manifest['changed_entries_total']
    report['all_checks_pass'] = bool(all_ok and report['changed_entries_total_computed'] == manifest['changed_entries_total'])
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    for e in report['tensors']:
        print(f"{e['name']}: base {'OK' if e['base_match'] else 'MISMATCH'}, edited {'OK' if e['edited_match'] else 'MISMATCH'}, "
              f"changed {e['changed_entries_computed']} (manifest {e['changed_entries_manifest']}), "
              f"cols {e['changed_columns_computed']} rows {e['changed_rows_computed']}")
    print('model file sha256', report['model_file_sha256'])
    print('ALL CHECKS PASS' if report['all_checks_pass'] else 'AUDIT FAILED')
    return 0 if report['all_checks_pass'] else 1


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:4]))

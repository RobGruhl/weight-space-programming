"""Independent, standard-library-only geometry check of the p50c delta (Claude).

For each edited tensor: rebuild the edited BF16 bits from the sparse patch, take the 64 changed columns of the
base (B) and edited (E) matrices as float32, form D = E - B, and measure
  - the rank-one fraction  sigma1^2 / ||D||_F^2  (power iteration, with one deflation step for sigma2),
  - the directional scale: with u the leading left singular vector, r_c = (u.E_c)/(u.B_c) per column, alpha = 1 - r,
  - the least-squares alpha* for the model D ~= -alpha u (u^T B) and its residual.
No numpy, torch, model load or GPU. Reads two tensors from the safetensors file by header offset.

    python3 -I geometry_stdlib.py <model.safetensors> <p50c edit dir> <report.json>
"""
import array
import ast
import hashlib
import json
import math
import struct
import sys
import zipfile
from pathlib import Path


def safetensors_header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def tensor_bytes(path, header, offset, name):
    info = header[name]
    a, b = info['data_offsets']
    with open(path, 'rb') as f:
        f.seek(offset + a)
        return info, f.read(b - a)


def read_npy(buf):
    if buf[:6] != b'\x93NUMPY':
        raise ValueError('not .npy')
    if buf[6] == 1:
        hlen, start = struct.unpack('<H', buf[8:10])[0], 10
    else:
        hlen, start = struct.unpack('<I', buf[8:12])[0], 12
    header = ast.literal_eval(buf[start:start + hlen].decode('latin1'))
    codes = {'<i8': 'q', '<u8': 'Q', '<i4': 'i', '<u4': 'I', '<i2': 'h', '<u2': 'H', '|u1': 'B'}
    arr = array.array(codes[header['descr']])
    arr.frombytes(buf[start + hlen:])
    return arr


def bf16_column(bits, rows, cols, c):
    """Column c of a row-major BF16 matrix as float32 values."""
    out = []
    for r in range(rows):
        out.append(struct.unpack('<f', struct.pack('<I', bits[r * cols + c] << 16))[0])
    return out


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def power_top_left(D_cols, rows, iters=200):
    """Leading left singular vector of the matrix whose columns are D_cols (each length rows)."""
    u = [0.0] * rows
    for col in D_cols:
        for i, x in enumerate(col):
            u[i] += x
    if norm(u) == 0:
        u = list(D_cols[0])
    u = [x / norm(u) for x in u]
    sigma = 0.0
    for _ in range(iters):
        coeffs = [dot(u, col) for col in D_cols]            # D^T u
        v = [0.0] * rows
        for col, c in zip(D_cols, coeffs):                  # D (D^T u)
            for i, x in enumerate(col):
                v[i] += c * x
        new_sigma = norm(coeffs)
        u = [x / norm(v) for x in v]
        if abs(new_sigma - sigma) < 1e-9 * max(1.0, new_sigma):
            sigma = new_sigma
            break
        sigma = new_sigma
    return u, sigma


def main(model_path, edit_dir, report_path):
    model_path, edit_dir = Path(model_path), Path(edit_dir)
    manifest = json.loads((edit_dir / 'manifest.json').read_text())
    npz = zipfile.ZipFile(edit_dir / 'edit.npz')
    header, offset = safetensors_header(model_path)
    report = {'audit': 'claude stdlib-only p50c geometry', 'tensors': []}
    for i, t in enumerate(manifest['tensors']):
        info, data = tensor_bytes(model_path, header, offset, t['name'])
        rows, cols = t['shape']
        base = array.array('H')
        base.frombytes(data)
        if hashlib.sha256(data).hexdigest() != t['base_sha256']:
            raise ValueError('base hash mismatch')
        idx, val = read_npy(npz.read(f't{i}_idx.npy')), read_npy(npz.read(f't{i}_val.npy'))
        edited = array.array('H', base)
        changed_cols = set()
        for j, v in zip(idx, val):
            if edited[j] != v:
                changed_cols.add(j % cols)
            edited[j] = v
        if hashlib.sha256(edited.tobytes()).hexdigest() != t['edited_sha256']:
            raise ValueError('edited hash mismatch')
        changed_cols = sorted(changed_cols)
        B = [bf16_column(base, rows, cols, c) for c in changed_cols]
        E = [bf16_column(edited, rows, cols, c) for c in changed_cols]
        D = [[e - b for e, b in zip(ec, bc)] for ec, bc in zip(E, B)]
        fro2 = sum(dot(col, col) for col in D)
        u, s1 = power_top_left(D, rows)
        # deflate once for sigma2
        coeffs = [dot(u, col) for col in D]
        D2 = [[x - c * ui for x, ui in zip(col, u)] for col, c in zip(D, coeffs)]
        _, s2 = power_top_left(D2, rows)
        ratios, ub, ud = [], [], []
        for bc, ec, dc in zip(B, E, D):
            a, b, d = dot(u, bc), dot(u, ec), dot(u, dc)
            ub.append(a)
            ud.append(d)
            if abs(a) > 1e-12:
                ratios.append(b / a)
        ratios_sorted = sorted(ratios)
        median_ratio = ratios_sorted[len(ratios_sorted) // 2]
        # least squares alpha* for D ~= -alpha u (u^T B): alpha* = -sum(ud*ub)/sum(ub^2)
        alpha_star = -sum(d * a for d, a in zip(ud, ub)) / sum(a * a for a in ub)
        resid2 = 0.0
        for col, a in zip(D, ub):
            pred = [-alpha_star * a * ui for ui in u]
            resid2 += sum((x - p) ** 2 for x, p in zip(col, pred))
        # how much of B's column mass lay along u before the edit
        b_along_u = sum(a * a for a in ub) / sum(dot(bc, bc) for bc in B)
        entry = {
            'name': t['name'],
            'changed_columns': len(changed_cols),
            'delta_frobenius': math.sqrt(fro2),
            'sigma1': s1,
            'sigma2': s2,
            'rank_one_fraction': s1 * s1 / fro2,
            'sigma2_over_sigma1': s2 / s1 if s1 else None,
            'median_column_ratio_u_E_over_u_B': median_ratio,
            'min_column_ratio': ratios_sorted[0],
            'max_column_ratio': ratios_sorted[-1],
            'alpha_from_median_ratio': 1 - median_ratio,
            'alpha_least_squares': alpha_star,
            'rank_one_model_relative_residual': math.sqrt(resid2 / fro2),
            'base_column_mass_fraction_along_u': b_along_u,
            'u_top5_abs_coords': sorted(((abs(x), k) for k, x in enumerate(u)), reverse=True)[:5],
        }
        report['tensors'].append(entry)
        print(json.dumps({k: v for k, v in entry.items() if k != 'u_top5_abs_coords'}, indent=1))
    Path(report_path).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:4])

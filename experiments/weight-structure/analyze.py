"""CPU-only geometry of public p50c BF16 edit. No inference or external services."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='2'
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parents[2]
MODEL=ROOT/'private/tiny-replay/model/model.safetensors'
PATCH=ROOT/'references/upstream/exp8_math/runs/p50c/edit/edit.npz'
OUT=ROOT/'experiments/weight-structure'
RAW=ROOT/'private/weight-structure'
RAW.mkdir(exist_ok=True)
report={'method':'CPU float64 SVD on BF16-realized selected-column deltas; direction sign fixed by largest absolute coordinate','patch_sha256':hashlib.sha256(PATCH.read_bytes()).hexdigest(),'layers':[]}
arrays={}; directions=[]
z=np.load(PATCH,allow_pickle=False)
with safe_open(MODEL,framework='pt',device='cpu') as f:
    for i,layer in enumerate([15,19]):
        name=f'model.language_model.layers.{layer}.mlp.down_proj.weight'
        w=f.get_tensor(name).float().numpy().astype(np.float64)
        edited=w.copy()
        edited.ravel()[z[f't{i}_idx']]=(z[f't{i}_val'].astype(np.uint32)<<16).view(np.float32)
        delta=edited-w
        cols=np.flatnonzero(np.any(delta!=0,axis=0))
        d=delta[:,cols]
        u,s,vh=np.linalg.svd(d,full_matrices=False)
        direction=u[:,0]*np.sign(u[np.argmax(np.abs(u[:,0])),0])
        directions.append(direction)
        p=direction@w[:,cols]; q=direction@edited[:,cols]
        alpha=-np.dot(p,direction@d)/np.dot(p,p)
        pred=-6*np.outer(direction,p)
        # Frozen negative control: row permutation of the realized update.
        rng=np.random.default_rng(20261107+layer)
        permutation=rng.permutation(1024)
        proposed=d[permutation,:]
        control=torch.from_numpy(w[:,cols]+proposed).to(torch.bfloat16).float().numpy().astype(np.float64)-w[:,cols]
        # New minimal intervention: sign reverse the realized patch, not a fitted reconstruction.
        reverse=torch.from_numpy(w[:,cols]-d).to(torch.bfloat16).float().numpy().astype(np.float64)-w[:,cols]
        cn=np.linalg.norm(d,axis=0)
        report['layers'].append({'layer':layer,'tensor':name,'changed_entries':int(np.count_nonzero(d)),'selected_columns':cols.tolist(),'weight_fro':float(np.linalg.norm(w)),'delta_fro':float(np.linalg.norm(d)),'relative_fro':float(np.linalg.norm(d)/np.linalg.norm(w)),'selected_relative_fro':float(np.linalg.norm(d)/np.linalg.norm(w[:,cols])),'singular_values':s.tolist(),'top1_energy':float(s[0]**2/s.dot(s)),'rank99':int(np.searchsorted(np.cumsum(s*s)/s.dot(s),.99)+1),'rank9999':int(np.searchsorted(np.cumsum(s*s)/s.dot(s),.9999)+1),'rank_float64':int(np.linalg.matrix_rank(d)),'alpha_fit':float(alpha),'alpha6_relative_residual':float(np.linalg.norm(d-pred)/np.linalg.norm(d)),'projection_ratio':float(p@q/(p@p)),'per_column':[{'column':int(c),'delta_norm':float(cn[j]),'weight_norm':float(np.linalg.norm(w[:,c])),'before_projection':float(p[j]),'after_projection':float(q[j]),'projection_ratio':float(q[j]/p[j])} for j,c in enumerate(cols)],'control':{'seed':20261107+layer,'direction_abs_cosine':float(abs(direction@direction[permutation])),'realized_norm_ratio':float(np.linalg.norm(control)/np.linalg.norm(d)),'max_per_column_norm_relative_drift':float(np.max(abs(np.linalg.norm(control,axis=0)/cn-1))),'changed_entries':int(np.count_nonzero(control))},'sign_reverse':{'realized_norm_ratio':float(np.linalg.norm(reverse)/np.linalg.norm(d)),'projection_ratio':float(p@(direction@(w[:,cols]+reverse))/(p@p))}})
        for key,value in {'columns':cols,'direction':direction,'singular_values':s,'delta':d,'permutation':permutation,'control_delta':control,'reverse_delta':reverse}.items(): arrays[f'layer{layer}_{key}']=value
report['cross_layer_abs_direction_cosine']=float(abs(directions[0]@directions[1]))
report['selected_column_overlap']=len(set(report['layers'][0]['selected_columns'])&set(report['layers'][1]['selected_columns']))
np.savez(RAW/'geometry.npz',**arrays)
(OUT/'geometry.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='layers'},indent=2))
for x in report['layers']: print(json.dumps({k:v for k,v in x.items() if k not in ['singular_values','per_column','selected_columns']},indent=2))

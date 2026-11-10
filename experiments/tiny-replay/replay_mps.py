"""Bounded local synthetic smoke test, NOT benchmark reproduction.
Own adapter for the published sparse BF16 patch; never imports upstream code.
"""
import os
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
import json, hashlib, time, resource, platform, re
from pathlib import Path
import numpy as np
import torch
import transformers
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen3_5TextConfig, Qwen3_5ForCausalLM, LogitsProcessor, StoppingCriteria
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'private/tiny-replay'
BASE=OUT/'model'
EDIT=ROOT/'references/upstream/exp8_math/runs/p50c/edit'
PROMPTS=[('addition','What is 17 + 28?', '45'),('algebra','Solve 3x + 7 = 22 for x.', '5'),('geometry','A rectangle has length 8 and width 5. What is its area?', '40'),('control','What is 2 + 2?', '4')]
SUFFIX=' Please reason step by step, and put your final answer within \\boxed{}.'
def sha(b): return hashlib.sha256(b).hexdigest()
def bits(t): return t.detach().cpu().contiguous().view(torch.uint16).numpy()
def h(t): return sha(bits(t).tobytes())
class Presence(LogitsProcessor):
 def __init__(self,n): self.n=n
 def __call__(self,ids,scores):
  for row in range(ids.shape[0]): scores[row,torch.unique(ids[row,self.n:])]-=1.5
  return scores
class Bound(StoppingCriteria):
 def __init__(self,n): self.n=n; self.t=time.monotonic(); self.reason=None
 def __call__(self,ids,scores,**kwargs):
  a=ids[0,self.n:].tolist()
  if time.monotonic()-self.t>=60: self.reason='time_limit'
  if len(a)>=64 and a[-16:]==a[-32:-16]==a[-48:-32]==a[-64:-48]: self.reason='repeated_16_token_block_four_times'
  return self.reason is not None

def main():
 torch.set_num_threads(4); torch.set_num_interop_threads(1)
 manifest={'status':'preregistered','model':'Qwen/Qwen3.5-0.8B','revision':'2fc06364715b967f1860aea9cf38778875588b17','device':'mps','dtype':'source mixed precision: BF16 plus 36 FP32 linear-attention tensors','quantization':None,'threads':4,'seed':0,'max_new_tokens':256,'seconds_per_generation':60,'prompts':PROMPTS,'suffix':SUFFIX,'thinking':True,'temperature':0.6,'top_p':0.95,'top_k':20,'presence_penalty':1.5,'torch':torch.__version__,'transformers':transformers.__version__,'python':platform.python_version(),'deviations':['Transformers Apple MPS, not vLLM GPU','Synthetic problems, not MATH-500/GSM8K/IFEval','256 tokens, not 32768; one sample per problem','60-second and exact repeated-block stop rules','Seed parity does not imply runtime sample parity'],'patch_archive_sha256':sha((EDIT/'edit.npz').read_bytes()),'adapter_sha256':sha(Path(__file__).read_bytes()),'files':{p.name:sha(p.read_bytes()) for p in BASE.iterdir() if p.is_file()}}
 (OUT/'run-manifest.json').write_text(json.dumps(manifest,indent=2))
 assert manifest['files']['model.safetensors']=='04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696'
 config=Qwen3_5TextConfig(**json.loads((BASE/'config.json').read_text())['text_config'])
 config._attn_implementation='eager'
 torch.set_default_dtype(torch.bfloat16)
 model=Qwen3_5ForCausalLM(config)
 torch.set_default_dtype(torch.float32)
 with safe_open(BASE/'model.safetensors',framework='pt',device='cpu') as f:
  state={k.replace('model.language_model.','model.',1):f.get_tensor(k) for k in f.keys() if k.startswith('model.language_model.')}
 if 'lm_head.weight' not in state: state['lm_head.weight']=state['model.embed_tokens.weight']
 model.load_state_dict(state,strict=True,assign=True); model.eval()
 assert all(p.dtype in (torch.bfloat16, torch.float32) for p in model.parameters())
 del state
 model.to('mps')
 tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True,trust_remote_code=False)
 patch=json.loads((EDIT/'manifest.json').read_text()); z=np.load(EDIT/'edit.npz',allow_pickle=False)
 before={k:h(v) for k,v in model.state_dict().items()}
 integrity=[]
 results=[]
 for variant in ['baseline','p50c']:
  if variant=='p50c':
   with torch.no_grad():
    for i,t in enumerate(patch['tensors']):
     assert t['form']=='sparse'
     name=t['name'].replace('model.language_model.','model.',1); tensor=model.state_dict()[name]
     assert h(tensor)==t['base_sha256']; assert list(tensor.shape)==t['shape']
     original=bits(tensor).copy(); edited=original.copy(); edited.reshape(-1)[z[f't{i}_idx']]=z[f't{i}_val']
     assert sha(edited.tobytes())==t['edited_sha256']; count=int(np.count_nonzero(edited!=original)); assert count==t['changed_entries']
     tensor.copy_(torch.from_numpy(edited).view(torch.bfloat16))
     assert h(tensor)==t['edited_sha256']
     integrity.append({'name':name,'base_sha256':t['base_sha256'],'edited_sha256':h(tensor),'changed_entries':count})
   changed={k for k,v in model.state_dict().items() if h(v)!=before[k]}
   assert changed=={x['name'] for x in integrity}
   (OUT/'integrity.json').write_text(json.dumps({'verified':True,'unexpected_changes':[],'tensors':integrity},indent=2))
  for ident,question,answer in PROMPTS:
   torch.manual_seed(0)
   prompt=tokenizer.apply_chat_template([{'role':'user','content':question+SUFFIX}],tokenize=False,add_generation_prompt=True,enable_thinking=True)
   inputs=tokenizer(prompt,return_tensors='pt').to('mps'); n=inputs.input_ids.shape[1]; stop=Bound(n); start=time.monotonic()
   with torch.inference_mode():
    ids=model.generate(**inputs,max_new_tokens=256,do_sample=True,temperature=0.6,top_p=0.95,top_k=20,logits_processor=[Presence(n)],stopping_criteria=[stop],pad_token_id=tokenizer.eos_token_id,use_cache=True)
   tokens=ids[0,n:]; text=tokenizer.decode(tokens,skip_special_tokens=False); boxes=re.findall(r'\\boxed\{([^{}]*)\}',text)
   row={'variant':variant,'id':ident,'expected':answer,'output':text,'output_ids':tokens.tolist(),'prompt_sha256':sha(prompt.encode()),'prompt_ids':inputs.input_ids[0].tolist(),'tokens':len(tokens),'seconds':time.monotonic()-start,'stop':stop.reason or ('token_limit' if len(tokens)==256 else 'eos'),'last_box':boxes[-1] if boxes else None,'correct':bool(boxes and boxes[-1].strip()==answer),'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
   row['exact_box_match']=row['correct']
   if row['stop']!='eos': row.update(grading_status='truncated_ungraded',correct=None)
   elif not row['last_box'] or not row['last_box'].strip(): row.update(grading_status='missing_answer_ungraded',correct=None)
   else: row['grading_status']='graded_completed_box'
   results.append(row); (OUT/'results.json').write_text(json.dumps(results,indent=2)); print(json.dumps({k:v for k,v in row.items() if k not in ['output','output_ids','prompt_ids']}),flush=True)
 manifest['status']='complete'; manifest['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; (OUT/'run-manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__': main()

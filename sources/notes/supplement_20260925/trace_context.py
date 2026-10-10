"""Recover original datastore rows only after exact full value-stream verification."""
from pathlib import Path
import csv,json,random,sys,bisect
import numpy as np
P=Path('/root/autodl-tmp/dev/第二篇_跨领域低资源方法迁移');S=P/'notes/supplement_20260925'
sys.path.insert(0,str(S/'code/e1'))
from common import norm
from provenance import atomic_json,sha256
from transformers import WhisperTokenizer
DS=Path('/root/autodl-tmp/e3_ds/e12_ksc2_lib1');R=P/'results/supplement_20260925'
events=[json.loads(l) for name in ['lib1_seed0.json.trace.jsonl','bare_trace.json.trace.jsonl'] for l in (R/name).open()]
needed=sorted({i for e in events for i in e.get('neighbor_ids',[])})
assert needed and min(needed)>=0
metas=[json.loads((DS/f'meta_s{i}.json').read_text()) for i in [0,1]]
assert all(m['seed']==0 and m['nshards']==2 and m['pool']==metas[0]['pool'] for m in metas)
pool=Path(metas[0]['pool']);rows=list(csv.DictReader(pool.open()))
random.Random(0).shuffle(rows);rows.sort(key=lambda r:float(r['duration']))
tok=WhisperTokenizer.from_pretrained(metas[0]['model'])
vals=np.load(DS/'vals_all.npy',mmap_mode='r');offset=0;mapping={};shard_offset=0
# Validate shard concatenation independently from reconstructed tokenization.
for f in sorted(DS.glob('vals_shard*.npy')):
 v=np.load(f,mmap_mode='r');assert np.array_equal(vals[shard_offset:shard_offset+len(v)],v);shard_offset+=len(v)
assert shard_offset==len(vals)
for shard in [0,1]:
 start=offset;picked=rows[shard::2];assert len(picked)==metas[shard]['utts']
 for j,r in enumerate(picked):
  text=norm(r.get('sentence') or r.get('wrd',''));tokens=tok.encode(text,add_special_tokens=False)[:400]
  assert np.array_equal(vals[offset:offset+len(tokens)],np.asarray(tokens,dtype=np.int32)),f'value reconstruction mismatch: shard={shard}, row={j}'
  lo=bisect.bisect_left(needed,offset);hi=bisect.bisect_left(needed,offset+len(tokens))
  for index in needed[lo:hi]:
   position=index-offset;mapping[str(index)]=dict(file_path=r['file_path'],manifest_text=text,token_position=position,
     prefix_token_ids=tokens[:position],prefix=tok.decode(tokens[:position]),value_token_id=tokens[position],value_text=tok.decode([tokens[position]]),construction_shard=shard)
  offset+=len(tokens)
 assert offset-start==metas[shard]['entries']
assert offset==len(vals) and len(mapping)==len(needed)
for e in events:
 for i,v in zip(e.get('neighbor_ids',[]),e.get('value_token_ids',[])):assert mapping[str(i)]['value_token_id']==v
atomic_json(S/'trace_contexts.json',dict(state='complete',verified_entries=offset,pool_sha256=sha256(pool),values_sha256=sha256(DS/'vals_all.npy'),neighbors=mapping,
 limitation='Reconstructed from recorded builder ordering and verified against every saved value; source key vectors were not regenerated. Actual retrieval IDs/queries are recorded by the decoder.'))
print('verified',offset,'entries;',len(mapping),'retrieved contexts')

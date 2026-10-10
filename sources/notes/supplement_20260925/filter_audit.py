from pathlib import Path
import csv, json, sys, math
P=Path('/root/autodl-tmp/dev/第二篇_跨领域低资源方法迁移');S=P/'notes/supplement_20260925'
sys.path.insert(0,str(S/'code/e1'))
from common import compression_ratio
from provenance import atomic_json,sha256
ROOT=Path('/root/autodl-tmp/data/final_manifests')
pairs={'kk':(ROOT/'e12_ksc2_pool_full.csv',ROOT/'ksc2_lib1_pseudo.csv'),
 'mn':(P/'results/e12_mnc_pool.csv',P/'results/e12_mnc_pool_filtered.csv'),
 'sw':(P/'results/e12_fleurs_sw_train_pseudo.csv',P/'results/e12_fleurs_sw_train_filtered.csv'),
 'tr':(P/'results/tr_repair_v3_20260925/pool.csv',P/'results/tr_repair_v3_20260925/filtered.csv')}
out={}
for lang,(source,filtered) in pairs.items():
 if not source.exists() or not filtered.exists():
  out[lang]={'state':'pending','missing':[str(x) for x in [source,filtered] if not x.exists()]};continue
 try:
  before=(sha256(source),sha256(filtered));names=['input','nonempty','alp_gt_minus1','CR_le_2.4','characters_per_second_le20','top80_after_gates']
  tally={n:dict(n=0,hours=0.) for n in names};eligible=[];seen=set()
  independent=dict(empty=0,low_alp=0,high_CR=0,high_char_rate=0)
  with source.open() as f:
   for r in csv.DictReader(f):
    path=r['file_path'];assert path not in seen;seen.add(path)
    dur=float(r['duration']);alp=float(r['avg_logprob']);assert math.isfinite(dur) and dur>0 and math.isfinite(alp)
    text=r['wrd'];cr=compression_ratio(text);rate=len(text)/max(.1,dur)
    gates=[bool(text.strip()),alp>-1,cr<=2.4,rate<=20]
    for k,ok in zip(independent,gates):independent[k]+=not ok
    tally['input']['n']+=1;tally['input']['hours']+=dur/3600
    for name,ok in zip(names[1:5],gates):
     if not ok:break
     tally[name]['n']+=1;tally[name]['hours']+=dur/3600
    if all(gates):eligible.append((alp,path,dur))
  eligible.sort(key=lambda x:-x[0]);kept=eligible[:int(len(eligible)*.8)];expected={r[1] for r in kept}
  tally[names[-1]]=dict(n=len(kept),hours=sum(r[2] for r in kept)/3600)
  with filtered.open() as f:actual=[r['file_path'] for r in csv.DictReader(f)]
  assert len(set(actual))==len(actual)
  assert before==(sha256(source),sha256(filtered)),'input changed during audit'
  actual_set=set(actual)
  for i,n in enumerate(names):
   tally[n]['retained_percent_input']=100*tally[n]['n']/tally['input']['n']
   if i:tally[n]['removed_since_previous']=tally[names[i-1]]['n']-tally[n]['n']
  out[lang]=dict(state='complete' if actual_set==expected else 'recipe_mismatch',source=str(source),filtered=str(filtered),source_sha256=before[0],filtered_sha256=before[1],sequential=tally,
    independent_failures=independent,actual_n=len(actual),missing_from_actual=len(expected-actual_set),extra_in_actual=len(actual_set-expected),note='Independent failure counts overlap. Quantile after hard gates; units characters/sec. Coverage separately checked by language audit.')
 except Exception as e:out[lang]=dict(state='failed',error=repr(e))
atomic_json(S/'filter_attrition.json',out)
print(json.dumps(out,ensure_ascii=False,indent=2))

"""CPU-only audit of complete saved hypotheses; no decoding or model loading."""
from pathlib import Path
import argparse,csv,datetime,json,math,re,sys,os,time,traceback
from collections import Counter
from functools import lru_cache
import numpy as np
import jiwer
from num2words import num2words
P=Path('/root/autodl-tmp/dev/第二篇_跨领域低资源方法迁移');S=P/'notes/cpu_analysis_20260925'
CODE=P/'notes/repair_20260925/code'
sys.path[:0]=[str(CODE/'e1'),str(CODE/'e12')]
from common import load_test,norm,compression_ratio
from common_cv import load_cv,load_fleurs,CV_ROOT
from provenance import atomic_json,sha256
NAMES=['unseen','freq1-5','freq6-100','freq_gt100']
sys.path.insert(0,str(P/'code/e3'))
from numeric_surface import convert_token, VERSION as NUMERIC_VERSION
@lru_cache(maxsize=50000)
def number(token,lang):
 return convert_token(token,lang)
def numtext(text,lang):return ' '.join(number(x,lang) for x in text.split())
def score(ref,hyp):
 m=jiwer.process_words(ref,hyp)
 return m,np.array([m.substitutions,m.deletions,m.insertions],dtype=np.int64)
def agg(ix,counts,words,cr,chars=None):
 ix=np.asarray(ix,dtype=int);n=int(words[ix].sum());sd=counts[ix].sum(0)
 r=dict(n_utterances=len(ix),reference_words=n,S=int(sd[0]),D=int(sd[1]),I=int(sd[2]),WER=100*float(sd.sum())/n if n else None,
  deletion_rate=100*float(sd[1])/n if n else None,insertion_rate=100*float(sd[2])/n if n else None,
  high_CR_count=int((cr[ix]>=2.4).sum()),high_CR_percent=100*float((cr[ix]>=2.4).mean()) if len(ix) else None)
 if chars is not None:
  ec,nc=chars[ix].sum(0);r['CER']=100*float(ec)/int(nc) if nc else None
 return r
def boot(diff,words,groups):
 da=np.array([diff[g].sum() for g in groups]);dn=np.array([words[g].sum() for g in groups]);rng=np.random.default_rng(0);draws=[]
 for _ in range(40):
  ix=rng.integers(0,len(groups),(250,len(groups)));draws.extend((100*da[ix].sum(1)/dn[ix].sum(1)).tolist())
 return dict(delta_pp=float(100*diff.sum()/words.sum()),ci95_pp=np.percentile(draws,[2.5,97.5]).tolist(),B=10000,seed=0,clusters=len(groups),interpretation='Exploratory marginal percentile CI; no adjusted confirmatory significance claim')
def clusters(lang,dataset,uids):
 if dataset=='ksc2':return [list(range(i,min(i+50,len(uids)))) for i in range(0,len(uids),50)],'lexical UID blocks of 50; not speaker IDs'
 if lang=='sw':return [list(range(i,min(i+50,len(uids)))) for i in range(0,len(uids),50)],'FLEURS row blocks of 50; only about 10 blocks, uncertainty estimates fragile'
 with (Path(CV_ROOT)/lang/'test.tsv').open() as f:cid={r['path']:r['client_id'] for r in csv.DictReader(f,delimiter='\t',quoting=csv.QUOTE_NONE)}
 gs={}
 for i,u in enumerate(uids):assert cid[u];gs.setdefault(cid[u],[]).append(i)
 return list(gs.values()),'CV client_id clusters'
def analyze(dataset,lang,data,paths,freqpath):
 refs={u:r for u,w,r in data};uids=list(refs);assert len(uids)==len(data)
 words=np.array([len(refs[u].split()) for u in uids]);assert np.all(words>0)
 groups,unit=clusters(lang,dataset,uids)
 freq=json.loads(freqpath.read_text()) if freqpath and freqpath.exists() else None
 out=dict(n=len(uids),reference_words=int(words.sum()),cluster_unit=unit,conditions={},comparisons={},exploratory_interactions={},frequency_source=str(freqpath) if freq else None,
  frequency_sha256=sha256(freqpath) if freq else None)
 counts={};crs={};pers={};digit_counts={};subsets={'ref_words1-5':np.where(words<=5)[0],'ref_words6-10':np.where((words>5)&(words<=10))[0],
 'ref_words11-20':np.where((words>10)&(words<=20))[0],'ref_words_gt20':np.where(words>20)[0],
 'reference_has_digits':np.array([i for i,u in enumerate(uids) if re.search(r'\d',refs[u])],dtype=int)}
 # Frequency cutoffs derived once from reference words, without splitting ties.
 positive=np.array([freq.get(w,0) for r in refs.values() for w in r.split() if freq and freq.get(w,0)>0])
 cuts=sorted(set(int(x) for x in np.quantile(positive,[.25,.5,.75],method='nearest'))) if len(positive) else []
 for arm,path in paths.items():
  if not path.exists():out.setdefault('pending',[]).append(str(path));continue
  initial=sha256(path);d=json.loads(path.read_text());per=d['per_utt'];assert set(per)==set(refs),(dataset,arm,'incomplete UID coverage')
  sd=[];cc=[];cr=[];dc=[];digit_union=[];bins={x:[0,0] for x in NAMES};qbins={};fine={};empty=0
  for i,u in enumerate(uids):
   r=refs[u];h=per[u]['hyp'];assert norm(h,lang=lang)==h,(arm,u,'non-normalized hypothesis')
   if 'raw_hyp' in per[u]:assert norm(per[u]['raw_hyp'],lang=lang)==h
   m,c=score(r,h);sd.append(c);cm=jiwer.process_characters(r,h);cc.append([cm.substitutions+cm.deletions+cm.insertions,cm.hits+cm.substitutions+cm.deletions]);cr.append(compression_ratio(h));empty+=not bool(h.strip())
   digit_union.append(bool(re.search(r'\d',r+' '+h)))
   if lang in ['kk','tr']:
    rn,hn=numtext(r,lang),numtext(h,lang);_,cn=score(rn,hn);dc.append([int(cn.sum()),len(rn.split())])
   if freq:
    ws=r.split()
    for chunks in m.alignments:
     for chunk in chunks:
      for w in ws[chunk.ref_start_idx:chunk.ref_end_idx]:
       f=freq.get(w,0);e=int(chunk.type!='equal');key=NAMES[0 if f==0 else 1 if f<=5 else 2 if f<=100 else 3]
       bins[key][0]+=1;bins[key][1]+=e
       q='unseen' if f==0 else 'q'+str(int(np.searchsorted(cuts,f,side='left'))+1)
       for target,label in [(qbins,q),(fine,'unseen' if f==0 else '10^'+str(int(math.log10(f))))]:
        v=target.setdefault(label,[0,0]);v[0]+=1;v[1]+=e
  sd=np.array(sd);cc=np.array(cc);cr=np.array(cr);counts[arm]=sd;crs[arm]=cr;pers[arm]=per
  summary=agg(range(len(uids)),sd,words,cr,cc);assert round(summary['WER'],2)==d['summary']['WER'];assert len(uids)==d['summary']['n']
  assert sha256(path)==initial,'file changed during scoring'
  summary.update(empty_count=empty,source=str(path),sha256=initial)
  high=sd.sum(1)>words
  summary['utterance_WER_above100']=dict(count=int(high.sum()),percent=100*float(high.mean()),error_share_percent=100*float(sd[high].sum())/sd.sum(),corpus_WER_contribution_pp=100*float(sd[high].sum())/words.sum(),interpretation='error-heavy utterances, not a hallucination classifier')
  row=dict(summary=summary,length_and_reference_digit_groups={k:agg(ix,sd,words,cr,cc) for k,ix in subsets.items()})
  row['digit_incidence']=dict(reference_count=len(subsets['reference_has_digits']),reference_or_this_hypothesis_count=sum(digit_union),note='Hypothesis-dependent incidence is descriptive; paired subgroup analysis uses reference-only membership.')
  if dc:
   dc=np.array(dc);digit_counts[arm]=dc
   row['integer_surface_sensitivity']=dict(WER=100*float(dc[:,0].sum())/int(dc[:,1].sum()),reference_words=int(dc[:,1].sum()),errors=int(dc[:,0].sum()),rule='Both sides: standalone ASCII integer tokens <=9 digits, no leading zeros; num2words kz/tr, Turkish output segmented; no mixed tokens, ordinals, dates, decimals or morphology handling. Exploratory secondary metric, not replacement scoring.')
  else:row['integer_surface_sensitivity']=dict(state='not_computed',reason='No validated converter for this language; digit incidence still reported. Legacy handwritten Mongolian converter excluded.')
  if freq:
   assert sum(v[0] for v in bins.values())==int(words.sum());assert sum(v[1] for v in bins.values())==int(sd[:,:2].sum())
   def fmt(b):return {k:dict(N=v[0],SD=v[1],SD_rate=100*v[1]/v[0] if v[0] else None) for k,v in b.items()}
   row['frequency']=dict(fixed=fmt(bins),reference_weighted_quantiles=fmt(qbins),quantile_cutoffs=cuts,log10_bins=fmt(fine),insertions_separate=int(sd[:,2].sum()),metric='(S+D)/N; not WER; quantile ties kept together')
  out['conditions'][arm]=row
  print(dataset,arm,'WER',round(summary['WER'],4),flush=True)
 if 'base' in counts:
  b=counts['base'];be=b.sum(1);fixed_high=np.where(crs['base']>=2.4)[0];tail=np.argsort(-be,kind='stable')[:max(1,math.ceil(.01*len(uids)))];regular=np.where(be<=words)[0]
  for arm,a in counts.items():
   if arm=='base':continue
   diff=a.sum(1)-be
   row=dict(delta=boot(diff,words,groups),win=int((diff<0).sum()),tie=int((diff==0).sum()),regress=int((diff>0).sum()),
    errors_removed=int(-diff[diff<0].sum()),errors_added=int(diff[diff>0].sum()),
    component_change={name:boot(a[:,j]-b[:,j],words,groups) for j,name in enumerate(['S','D','I'])},
    fixed_subsets={label:dict(base=agg(ix,b,words,crs['base']),condition=agg(ix,a,words,crs[arm]),delta_pp=100*float(diff[ix].sum())/words[ix].sum() if words[ix].sum() else None) for label,ix in dict(subsets,baseline_high_CR=fixed_high,baseline_top1pct_error_count=tail,baseline_WER_le100=regular).items()})
   row['illustrative_extremes']={label:[dict(uid=uids[i],ref=refs[uids[i]],base_hyp=pers['base'][uids[i]]['hyp'],hyp=pers[arm][uids[i]]['hyp'],delta_errors=int(diff[i])) for i in np.argsort(sign*diff,kind='stable')[:5]] for label,sign in [('improvement',1),('regression',-1)]}
   if arm in digit_counts and 'base' in digit_counts:
    da,db=digit_counts[arm],digit_counts['base'];assert np.array_equal(da[:,1],db[:,1]);row['integer_surface_delta']=boot(da[:,0]-db[:,0],da[:,1],groups)
   out['comparisons'][arm]=row
 if all(x in counts for x in ['lib1','lib1f','lib4','lib4f']):
  diff=counts['lib1'].sum(1)-counts['lib1f'].sum(1)-counts['lib4'].sum(1)+counts['lib4f'].sum(1)
  out['exploratory_interactions']['filter_by_transcript_source']=dict(formula='(pseudo filtered - pseudo full) - (gold filtered - gold full); tuned configurations, no pure-size control',**boot(diff,words,groups))
 if 'beam5' in counts:
  out['standard_comparisons']={arm:boot(counts[arm].sum(1)-counts['beam5'].sum(1),words,groups) for arm in ['lib1','lib3p']}
 if all(x in counts for x in ['base','lib1','bare_base','bare_lib1']):
  out['exploratory_interactions']['retrieval_by_protection']=boot(counts['lib1'].sum(1)-counts['base'].sum(1)-counts['bare_lib1'].sum(1)+counts['bare_base'].sum(1),words,groups)
 return out
def run():
 S.mkdir(exist_ok=True);atomic_json(S/'status.json',dict(phase='running',time=datetime.datetime.now().isoformat(),pid=os.getpid()))
 arms={'base':'0.0','lib1':'0.4','lib1f':'0.4','lib3p':'0.4','lib2':'0.4','lib4':'0.5','lib4f':'0.5'}
 freq=Path('/root/autodl-tmp/e3_ds/e12_ksc2_lib1/word_freq.json')
 out=dict(created=datetime.datetime.now().isoformat(),scope='CPU only; saved complete hypotheses, no model loading, no GPU experiment, exploratory diagnostics',datasets={},pending={},errors={},reused_verified={})
 jobs=[]
 for name,suffix,data in [('ksc2','test',load_test()),('cvkk','cvtest',load_cv('kk','test'))]:
  paths={a:P/'results/kk_matrix'/f'kkm_{a}_{suffix}_lam{lam}.json' for a,lam in arms.items()}
  if name=='ksc2':paths.update({a:P/'results'/f for a,f in {'greedy':'e1std_greedy_test.json','beam5':'e1std_beam5_test.json','bare_base':'kkm_base_test_lam0.0_bare.json','bare_lib1':'kkm_lib1_test_lam0.4_nostack.json','permuted':'kkm_lib1rv_test_lam0.4.json'}.items()})
  jobs.append((name,'kk',data,paths,freq))
 # Only promote fresh cross-language results after the existing strict audit.
 swpaths={'base':P/'results/e12_fleurs_reverify_sw_lam0.json'}
 audit_path=P/'notes/mn_sw_review_20260925/audit.json';audit=json.loads(audit_path.read_text()) if audit_path.exists() else {}
 good=audit.get('outputs',{})
 for arm,f in [('memory_fixed05','e12_fleurs_reverify_sw_lam05.json'),('memory_diagnostic03','e12_fleurs_reverify_sw_lam03.json')]:
  if f in good and 'e12_fleurs_sw_train_filtered.csv' in good:swpaths[arm]=P/'results'/f
  else:out['pending']['sw_'+arm]='Await full saved output and pool audit; no test-based selection.'
 jobs.append(('sw','sw',load_fleurs('sw_ke','test'),swpaths,Path('/root/autodl-tmp/e3_ds/e12_fleurs_sw/word_freq.json') if 'e12_fleurs_sw_train_filtered.csv' in good else None))
 best=audit.get('checks',{}).get('mn_lambda_star_exact');mnbase='e12_mnc_test_v2_lam00.json'
 mnname=f'e12_mnc_test_v2_lam{best:g}.json' if best is not None else ''
 if mnbase in good and mnname in good and 'e12_mnc_pool_filtered.csv' in good and not any('mn' in e for e in audit.get('errors',[])):
  jobs.append(('mn','mn',load_cv('mn','test'),{'base':P/'results'/mnbase,'memory':P/'results'/mnname},Path('/root/autodl-tmp/e3_ds/e12_mnc/word_freq.json')))
 else:out['pending']['mn']='Await clean pool, complete dev selection and full test acceptance.'
 tr=P/'results/tr_repair_v3_20260925';st=json.loads((tr/'status.json').read_text())
 if st['phase']=='complete':jobs.append(('tr','tr',load_cv('tr','test'),{'base':tr/'test_baseline.json','memory':tr/'test_retrieval.json'},Path('/root/autodl-tmp/e3_ds/tr_repair_v3_20260925/word_freq.json')))
 else:out['pending']['tr']='Await corrected full rerun completion; old normalized hypotheses excluded.'
 for args in jobs:
  try:
   dataset,lang,data,paths,freqpath=args
   fingerprint=dict(code_sha256=sha256(Path(__file__)),converter_sha256=sha256(P/'code/e3/numeric_surface.py'),sources={str(p):sha256(p) if p.exists() else None for p in paths.values()},
    references_sha256=__import__('hashlib').sha256(json.dumps(data,ensure_ascii=False).encode()).hexdigest(),frequency_sha256=sha256(freqpath) if freqpath and freqpath.exists() else None)
   cache=S/f'{dataset}.cache.json';artifact=S/f'{dataset}.json'
   if cache.exists() and artifact.exists() and json.loads(cache.read_text())==fingerprint:
    result=json.loads(artifact.read_text());print(dataset,'reused unchanged verified analysis',flush=True)
   else:
    result=analyze(*args);atomic_json(artifact,result);atomic_json(cache,fingerprint)
   out['datasets'][dataset]=result
  except Exception:out['errors'][args[0]]=traceback.format_exc()
 for f in ['notes/revision_08_overlap_audit_20260925.json','notes/revision_11_statistics_audit_20260925.json','notes/revision_11_implementation_audit_20260925.json','notes/experiment_review_20260925/new_cpu_contrasts.json','notes/supplement_20260925/filter_attrition.json']:
  q=P/f
  if q.exists():out['reused_verified'][f]=dict(sha256=sha256(q),note='Existing audited overlap, full-dev grid, six-contrast Holm, implementation, fixed-lambda, or attrition evidence; not recast as new experiments.')
 atomic_json(S/'report.json',out)
 with (S/'condition_metrics.csv').open('w',newline='') as f:
  fields=['dataset','condition','n_utterances','reference_words','WER','CER','S','D','I','deletion_rate','insertion_rate','high_CR_percent','empty_count'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for ds,d in out['datasets'].items():
   for arm,a in d['conditions'].items():writer.writerow({k:dict(dataset=ds,condition=arm,**a['summary']).get(k) for k in fields})
 atomic_json(S/'status.json',dict(phase='failed' if out['errors'] else 'available_results_complete',pending=out['pending'],errors=out['errors'],time=datetime.datetime.now().isoformat()))
 return out
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--after-language-chain',action='store_true');args=ap.parse_args()
 if args.after_language_chain:
  protocol=json.loads((P/'notes/supplement_20260925/protocol.json').read_text())
  def alive(x):
   try:
    t=Path(f"/proc/{x['pid']}/stat").read_text();f=t[t.rfind(')')+2:].split();return f[0]!='Z' and f[19]==x['starttime']
   except FileNotFoundError:return False
  atomic_json(S/'watch_status.json',dict(state='waiting_for_languages',pid=os.getpid(),dependencies=protocol['wait_for']))
  while any(alive(x) for x in protocol['wait_for']):time.sleep(30)
  import subprocess
  for script in [P/'notes/mn_sw_review_20260925/audit.py',P/'notes/supplement_20260925/filter_audit.py']:
   subprocess.run([sys.executable,str(script)],check=True)
 try:
  result=run()
  if args.after_language_chain:atomic_json(S/'watch_status.json',dict(state='failed' if result['errors'] else 'complete_with_pending' if result['pending'] else 'complete',errors=result['errors'],time=datetime.datetime.now().isoformat()))
 except Exception:
  atomic_json(S/'status.json',dict(phase='failed',error=traceback.format_exc()));raise

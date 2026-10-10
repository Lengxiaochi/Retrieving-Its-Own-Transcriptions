"""Strict saved-output scoring. Secondary surface metric, never a replacement for raw WER."""
from pathlib import Path
import argparse,csv,hashlib,json,os,sys
import jiwer
from numeric_surface import VERSION,normalize_text,normalize_numbers
ROOT=Path('/root/autodl-tmp/dev/第二篇_跨领域低资源方法迁移')
CV=Path('/root/autodl-tmp/data/CommonVoice/cv-corpus-26.0-2026-06-12')
KSC=Path('/root/autodl-tmp/share/ksc2/ISSAI_KSC2')
def load_refs(split,cvlang):
 refs={}
 if cvlang:
  with (CV/cvlang/f'{split}.tsv').open() as f:
   for row in csv.DictReader(f,delimiter='\t',quoting=csv.QUOTE_NONE):
    assert row.get('path') is not None and row.get('sentence') is not None,'Malformed TSV'
    ref=normalize_text(row['sentence'],cvlang)
    if not ref or not (CV/cvlang/'clips'/row['path']).is_file():continue
    assert row['path'] not in refs,'Duplicate UID';refs[row['path']]=ref
 else:
  raw=json.loads((KSC/f'{split}.json').read_text())
  for uid,row in sorted(raw.items()):
   path=row['transcript'].replace('/home/24_linhy/dataset/ksc2/ISSAI_KSC2/',str(KSC)+'/')
   refs[uid]=normalize_text(Path(path).read_text(),'kk')
 assert refs and all(refs.values())
 return refs
def evaluate(refs,artifact,lang):
 per=artifact['per_utt'];assert set(refs)==set(per),f'Incomplete UID coverage: missing={len(set(refs)-set(per))}, extra={len(set(per)-set(refs))}'
 assert artifact['summary']['n']==len(refs)
 rows={k:dict(errors=0,reference_words=0) for k in ['original','integer_surface']}
 for uid,ref in refs.items():
  row=per[uid];hyp=row['hyp'];assert normalize_text(hyp,lang)==hyp
  if lang=='tr':assert 'raw_hyp' in row,'Legacy Turkish normalized hypotheses cannot be repaired without raw_hyp; use repaired rerun'
  if 'raw_hyp' in row:assert normalize_text(row['raw_hyp'],lang)==hyp,'Normalization mismatch'
  if 'ref' in row:assert row['ref']==ref,'Saved/reference mismatch'
  for key,r,h in [('original',ref,hyp),('integer_surface',normalize_numbers(ref,lang),normalize_numbers(hyp,lang))]:
   m=jiwer.process_words(r,h);rows[key]['errors']+=m.substitutions+m.deletions+m.insertions;rows[key]['reference_words']+=m.hits+m.substitutions+m.deletions
 for row in rows.values():row['WER']=100*row['errors']/row['reference_words']
 assert round(rows['original']['WER'],2)==artifact['summary']['WER'],'Original score not reproduced'
 return dict(version=VERSION,n=len(refs),lang=lang,metrics=rows,
  interpretation='Secondary spelling sensitivity on already normalized text; neither semantic numeric accuracy nor corrected ground truth. No dates, decimals, ordinals, suffixes or alternate grammatical readings resolved.')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--hyp',required=True);ap.add_argument('--split',choices=['dev','test'],default='test');ap.add_argument('--cv-lang',choices=['kk','tr']);ap.add_argument('--out');args=ap.parse_args()
 path=Path(args.hyp);target=Path(args.out) if args.out else path.with_name(path.stem+'_digits_v2.json')
 if target.exists():raise FileExistsError(f'Refusing to overwrite {target}')
 refs=load_refs(args.split,args.cv_lang);d=json.loads(path.read_text());r=evaluate(refs,d,args.cv_lang or 'kk')
 r.update(input=str(path),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reference_sha256=hashlib.sha256(json.dumps(refs,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),converter_sha256=hashlib.sha256(Path(__file__).with_name('numeric_surface.py').read_bytes()).hexdigest())
 target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_name(target.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(r,ensure_ascii=False,indent=2));os.replace(tmp,target);print(json.dumps(r,ensure_ascii=False))
if __name__=='__main__':main()

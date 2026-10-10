#!/usr/bin/env python3
"""Explicit, inspectable entry points; no GPU job starts without --execute."""
from pathlib import Path
import argparse, hashlib, json, shlex, subprocess, sys, zipfile
ROOT = Path(__file__).resolve().parent
def read(name): return json.loads((ROOT/name).read_text(encoding='utf-8'))
def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('list'); sub.add_parser('verify'); sub.add_parser('demo')
    q=sub.add_parser('prepare'); q.add_argument('--mapping', default='paths.server.json')
    q=sub.add_parser('run'); q.add_argument('--execute', action='store_true'); q.add_argument('stage'); q.add_argument('args', nargs=argparse.REMAINDER)
    q=sub.add_parser('export'); q.add_argument('--out', required=True)
    a=p.parse_args(); stages=read('stages.json')
    if a.cmd=='list':
        for key,v in stages.items(): print(f"{key:18} {v['module']}\n  {v['script']}\n  {v['io']}")
    elif a.cmd=='verify':
        errors=[]; n=0
        for item in read('source_manifest.json'):
            f=ROOT/'sources'/item['file']
            if hashlib.sha256(f.read_bytes()).hexdigest()!=item['sha256']: errors.append(str(f))
            if f.suffix=='.py': compile(f.read_text(encoding='utf-8-sig'),str(f),'exec'); n+=1
        if errors: raise SystemExit('Hash mismatch: '+str(errors))
        print(f'PASS: source hashes; Python syntax {n} files. This is not GPU reproduction.')
    elif a.cmd=='demo':
        subprocess.run([sys.executable,str(ROOT/'tools/demo_metrics.py')],check=True)
    elif a.cmd=='prepare':
        target=ROOT/'runtime'
        if target.exists(): raise SystemExit('runtime already exists: preserve it or choose a fresh package directory.')
        mapping=read(a.mapping); mapping={k:v.replace('{runtime}',target.as_posix()) for k,v in mapping.items()}
        if any('<SET_' in v for v in mapping.values()): raise SystemExit('Fill in all <SET_...> values first.')
        audit=[]
        for f in (ROOT/'sources').rglob('*'):
            if not f.is_file(): continue
            content=f.read_text(encoding='utf-8'); dest=target/f.relative_to(ROOT/'sources')
            # One pass prevents a replacement target from being replaced again.
            import re
            pattern='|'.join(re.escape(k) for k in sorted(mapping,key=len,reverse=True))
            updated=re.sub(pattern,lambda m:mapping[m.group(0)],content) if pattern else content
            dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text(updated,encoding='utf-8')
            audit.append({'file':str(dest.relative_to(target)),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
        (target/'prepared.json').write_text(json.dumps({'mapping':mapping,'files':audit},ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'Prepared {len(audit)} source/config files in {target}. No model/data copied.')
    elif a.cmd=='run':
        if a.stage not in stages: raise SystemExit('Unknown stage; use list.')
        f=ROOT/'runtime'/stages[a.stage]['script']
        args=a.args[1:] if a.args[:1]==['--'] else a.args
        cmd=[sys.executable,str(f)]+args
        print(shlex.join(cmd),flush=True)
        if not a.execute: return
        if not f.exists(): raise SystemExit('Run prepare first.')
        if not args: raise SystemExit('Pass explicit stage arguments; see docs/运行说明.md.')
        subprocess.run(cmd,cwd=ROOT/'runtime',check=True)
    elif a.cmd=='export':
        dest=Path(a.out).resolve(); files=read('release_files.json')
        if dest.exists(): raise SystemExit('Output already exists; choose a new file.')
        dest.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
            for rel in files: z.write(ROOT/rel,rel)
        print(f'Exported {len(files)} allowlisted files to {dest}')
if __name__=='__main__': main()

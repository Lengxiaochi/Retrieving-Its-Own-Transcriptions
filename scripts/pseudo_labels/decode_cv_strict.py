"""CV pseudolabeling: strict TSV, isolated splits, raw outputs, resumable journal."""
import argparse, csv, hashlib, json, os, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'shared'))
from common import norm, NORMALIZATION_VERSION
from provenance import atomic_json, run_metadata
CV_ROOT = Path('/root/autodl-tmp/data/CommonVoice/cv-corpus-26.0-2026-06-12')
MODEL = '/root/autodl-tmp/dev/kazakh_asr/Whisper/model/whisper-large-v3-ct2'

def read_tsv(path):
    with path.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE))
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError(f'Malformed TSV: {path}')
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lang', default='tr')
    ap.add_argument('--split', choices=['train', 'pool'], default='train')
    ap.add_argument('--hours', type=float, default=45)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    assert args.hours > 0 and 0 <= args.shard < args.num_shards
    base = CV_ROOT / args.lang
    source_names = ['train'] if args.split == 'train' else ['validated', 'other']
    files = [base / f'{sp}.tsv' for sp in source_names + ['dev', 'test', 'clip_durations']]
    held_rows = read_tsv(base / 'dev.tsv') + read_tsv(base / 'test.tsv')
    held = {r['path'] for r in held_rows}
    held_spk = {r['client_id'] for r in held_rows if r['client_id']}
    rows, seen, source_counts, exclusions = [], set(), {}, {'held_clip':0, 'held_speaker':0, 'duplicate':0}
    for sp in source_names:
        source = read_tsv(base / f'{sp}.tsv')
        source_counts[sp] = len(source)
        if args.lang == 'tr' and sp == 'train':
            assert len(source) == 41401, source_counts
        for r in source:
            uid = r['path']
            reason = ('held_clip' if uid in held else 'held_speaker' if r['client_id'] in held_spk
                      else 'duplicate' if uid in seen else None)
            if reason:
                exclusions[reason] += 1
                continue
            if not uid or Path(uid).name != uid or not r['client_id']:
                raise ValueError(f'Invalid clip or unknown speaker: {uid!r}')
            seen.add(uid)
            rows.append(r)
    durations = {r['clip']:float(r['duration[ms]'])/1000 for r in read_tsv(base/'clip_durations.tsv')}
    random.Random(args.seed).shuffle(rows)
    picked, seconds = [], 0.
    for r in rows:
        uid = r['path']
        if durations.get(uid,0) <= 0 or not (base/'clips'/uid).is_file():
            raise ValueError(f'Missing audio or duration: {uid}')
        picked.append({'uid':uid, 'file_path':str(base/'clips'/uid),
                       'client_id':r['client_id'], 'duration':durations[uid]})
        seconds += durations[uid]
        if seconds >= args.hours*3600:
            break
    if seconds < args.hours*3600:
        raise ValueError(f'Insufficient isolated audio: {seconds/3600:.3f}h')
    if args.limit:
        picked = picked[:args.limit]
    picked = picked[args.shard::args.num_shards]
    assert picked and not ({r['uid'] for r in picked} & held)
    assert not ({r['client_id'] for r in picked} & held_spk)
    manifest = dict(source_counts=source_counts, exclusions=exclusions, selection_seed=args.seed,
                    rows=picked, hours=sum(r['duration'] for r in picked)/3600,
                    normalization_version=NORMALIZATION_VERSION)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(str(out)+'.manifest.json')
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, 'Frozen selection changed'
    else:
        atomic_json(manifest_path, manifest)
    print(f'{args.lang}: source={source_counts}, exclusions={exclusions}, selected={len(picked)}, {manifest["hours"]:.4f}h', flush=True)
    if args.prepare_only:
        return
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite output: {out}')
    metadata = run_metadata(args, inputs=files+[manifest_path, Path(MODEL)/'config.json'])
    metadata['model'] = MODEL
    metadata['transcribe'] = dict(language=args.lang, beam_size=1, vad_filter=False)
    meta_path = Path(str(out)+'.run.json')
    if meta_path.exists():
        assert json.loads(meta_path.read_text()) == metadata, 'Resume requires same code/config/inputs'
    else:
        atomic_json(meta_path, metadata)
    journal = Path(str(out)+'.raw.jsonl')
    done = {}
    if journal.exists():
        with journal.open(encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                assert row['uid'] not in done
                done[row['uid']] = row
    assert set(done) <= {r['uid'] for r in picked}
    import ctranslate2
    from faster_whisper import WhisperModel
    fw = WhisperModel(MODEL, device='cuda', compute_type='float16')
    start = time.time()
    with journal.open('a', encoding='utf-8') as f:
        for i,r in enumerate(picked):
            if r['uid'] in done:
                continue
            seed = int.from_bytes(hashlib.sha256(f'{args.seed}:{r["uid"]}'.encode()).digest()[:4], 'little') & 0x7fffffff
            ctranslate2.set_random_seed(seed)
            segments,_ = fw.transcribe(r['file_path'], **metadata['transcribe'])
            segments = list(segments)
            raw = ' '.join(s.text for s in segments)
            row = dict(r, raw_hyp=raw, wrd=norm(raw,lang=args.lang),
                       token_ids=[t for s in segments for t in s.tokens], seed=seed,
                       normalization_version=NORMALIZATION_VERSION,
                       avg_logprob=round(sum(s.avg_logprob for s in segments)/max(1,len(segments)),4) if segments else -9.9,
                       no_speech_prob=round(min((s.no_speech_prob for s in segments),default=1.),4))
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
            f.flush()
            done[r['uid']] = row
            if (i+1)%500 == 0:
                os.fsync(f.fileno())
                print(f'{len(done)}/{len(picked)} elapsed={(time.time()-start)/3600:.2f}h',flush=True)
    assert set(done) == {r['uid'] for r in picked}
    fields = ['file_path','wrd','duration','avg_logprob','no_speech_prob','raw_hyp','uid','client_id','normalization_version','seed']
    tmp = Path(str(out)+'.tmp')
    with tmp.open('w',encoding='utf-8',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(done[r['uid']] for r in picked)
    os.replace(tmp,out)
    print('DONE',out,len(done),flush=True)

if __name__ == '__main__':
    main()


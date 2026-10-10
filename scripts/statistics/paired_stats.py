"""Strict paired corpus-WER bootstrap; changes are percentage points / percent."""
import numpy as np
import jiwer

def validate_coverage(refs, per):
    expected, actual = set(refs), set(per)
    if expected != actual:
        raise ValueError(f'UID mismatch: missing={len(expected-actual)}, extra={len(actual-expected)}')
    if not expected or any(not r.strip() for r in refs.values()):
        raise ValueError('Empty reference set or reference')
    for uid,row in per.items():
        if 'ref' in row and row['ref'] != refs[uid]:
            raise ValueError(f'Normalization/reference mismatch: {uid}')

def paired(refs,a,b,clusters,B=10000,seed=0):
    validate_coverage(refs,a)
    validate_coverage(refs,b)
    units = [u for us in clusters.values() for u in us]
    if len(units)!=len(refs) or set(units)!=set(refs):
        raise ValueError('Clusters must partition all reference UIDs exactly once')
    ea,eb,ns = [],[],[]
    for us in clusters.values():
        if not us:
            raise ValueError('Empty cluster')
        ca=cb=n=0
        for u in us:
            ma,mb = (jiwer.process_words(refs[u],per[u]['hyp']) for per in (a,b))
            ca += ma.substitutions+ma.deletions+ma.insertions
            cb += mb.substitutions+mb.deletions+mb.insertions
            n += ma.hits+ma.substitutions+ma.deletions
        ea.append(ca); eb.append(cb); ns.append(n)
    ea,eb,ns = map(np.asarray,(ea,eb,ns))
    wa,wb = 100*ea.sum()/ns.sum(),100*eb.sum()/ns.sum()
    if B<100 or ea.sum()==0:
        raise ValueError('Need B>=100 and nonzero baseline errors for relative intervals')
    rng=np.random.default_rng(seed)
    absolute,relative=[],[]
    for _ in range(B):
        ix=rng.integers(0,len(ns),size=len(ns))
        den,base,diff=ns[ix].sum(),ea[ix].sum(),(eb[ix]-ea[ix]).sum()
        if base==0:
            raise ValueError('Relative change undefined in zero-error baseline resample')
        absolute.append(100*diff/den); relative.append(100*diff/base)
    absolute=np.asarray(absolute)
    delta=wb-wa
    p=(1+np.count_nonzero(np.abs(absolute-delta)>=abs(delta)))/(B+1)
    return dict(W_A=float(wa),W_B=float(wb),d_abs=float(delta),d_rel=float(100*delta/wa),
                ci_abs=np.percentile(absolute,[2.5,97.5]).tolist(),ci_rel=np.percentile(relative,[2.5,97.5]).tolist(),
                p_two=float(p),n_words=int(ns.sum()),n_utt=len(refs),n_clusters=len(ns),B=B,seed=seed,
                test='approximate null-centered paired bootstrap',
                interval='marginal percentile 95%; relative denominator resampled',
                practical_threshold_relative_percent=5.,practical_threshold_pass=bool(abs(100*delta/wa)>=5.))

def holm(results,family_size):
    if len(results)!=family_size:
        raise ValueError('Incomplete comparison family')
    running=0.
    for rank,(label,r) in enumerate(sorted(results.items(),key=lambda x:x[1]['p_two'])):
        running=min(1.,max(running,(family_size-rank)*r['p_two']))
        r.update(p_holm=running,holm_sig=running<.05)


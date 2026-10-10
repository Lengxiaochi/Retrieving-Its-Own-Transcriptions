#!/usr/bin/env python3
"""CPU teaching example only. Not a paper result."""
import math, statistics
def wer(s,d,i,n):
    if n<=0: raise ValueError('Reference word count must be positive')
    return 100*(s+d+i)/n
def main():
    print('教学示例（人工构造，非论文实验）：参考共 100 词')
    a=wer(10,4,6,100); b=wer(8,5,2,100)
    print(f'WER {a:.1f}% -> {b:.1f}%; ΔWER={b-a:+.1f} pp; 相对下降={(a-b)/a*100:.1f}%')
    print('S:10→8，D:4→5，I:6→2；总 WER 改善，但删除错误增加。')
    # LAMP A1 historical confusion matrix, not a new experiment.
    tp,fp,fn,tn=34538,6701,20197,43319
    print('LAMP A1 历史计数的公式演示（不是新测量）：')
    print(f'precision={tp/(tp+fp):.4%}; recall(S+I)={tp/(tp+fn):.4%}; FPR={fp/(fp+tn):.4%}')
    print(f'sample SD of rounded LAMP seed WER: {statistics.stdev([45.59,45.85,45.41]):.4f} pp')
if __name__=='__main__': main()


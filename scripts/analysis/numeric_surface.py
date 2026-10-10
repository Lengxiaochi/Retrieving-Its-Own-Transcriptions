"""Conservative integer-spelling sensitivity; deliberately not a semantic normalizer."""
import re,unicodedata
from functools import lru_cache
from num2words import num2words
VERSION='standalone-ascii-cardinals-kk-tr-v1-20260925'
SUPPORTED={'kk','tr'}
TR_ATOMS=sorted(['sıfır','bir','iki','üç','dört','beş','altı','yedi','sekiz','dokuz','on','yirmi','otuz','kırk','elli','altmış','yetmiş','seksen','doksan','yüz','bin','milyon'],key=len,reverse=True)
def normalize_text(text,lang):
 text=unicodedata.normalize('NFKC',str(text or ''))
 text=text.replace('I','ı').replace('İ','i') if lang=='tr' else text.replace('İ','I')
 text=text.lower()
 return ' '.join(''.join(c if c.isspace() or unicodedata.category(c).startswith(('L','M','N')) else ' ' for c in text).split())
@lru_cache(maxsize=50000)
def convert_token(token,lang):
 if lang not in SUPPORTED:raise ValueError(f'No validated integer-spelling converter for {lang}')
 if not re.fullmatch('[0-9]+',token) or len(token)>9 or (len(token)>1 and token[0]=='0'):return token
 text=num2words(int(token),lang='kz' if lang=='kk' else 'tr')
 if lang=='tr':
  text=text.replace(' ','');out=[]
  while text:
   match=next((w for w in TR_ATOMS if text.startswith(w)),None)
   if match is None:raise ValueError(f'Unsupported Turkish converter output for {token}: {text}')
   out.append(match);text=text[len(match):]
  text=' '.join(out)
 return normalize_text(text,lang)
def normalize_numbers(text,lang):
 if lang not in SUPPORTED:raise ValueError(f'No validated integer-spelling converter for {lang}')
 return ' '.join(convert_token(w,lang) for w in text.split())


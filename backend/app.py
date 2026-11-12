from __future__ import annotations
import csv, json, os, re, urllib.request, urllib.error
from pathlib import Path
from typing import Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE=Path(__file__).resolve().parents[1]; DATA=BASE/'data'; FRONTEND=BASE/'frontend'
CHUNK_WORDS=int(os.getenv('CHUNK_WORDS','110')); CHUNK_OVERLAP=int(os.getenv('CHUNK_OVERLAP','25'))
TOP_K=int(os.getenv('TOP_K','3')); MIN_RELEVANCE=float(os.getenv('MIN_RELEVANCE','0.08'))
app=FastAPI(title='Energy Operations Document RAG Assistant',version='1.0.0')
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_credentials=False,allow_methods=['GET','POST'],allow_headers=['*'])
class Query(BaseModel):
    text:str=Field(min_length=3,max_length=1000)
    top_k:int=Field(default=TOP_K,ge=1,le=5)
class Chunk:
    def __init__(self,source,title,text,index): self.source,self.title,self.text,self.index=source,title,text,index

def chunk_text(text,size=CHUNK_WORDS,overlap=CHUNK_OVERLAP):
    words=re.findall(r'\S+',text.strip())
    if not words:return []
    size=max(20,size); overlap=min(max(0,overlap),size-1); result=[]; start=0
    while start<len(words):
        end=min(start+size,len(words)); result.append(' '.join(words[start:end]))
        if end==len(words):break
        start=end-overlap
    return result

def load_docs():
    chunks=[]
    for p in sorted(DATA.glob('*.md')):
        raw=p.read_text(encoding='utf-8'); m=re.search(r'^#\s+(.+)$',raw,re.M)
        title=m.group(1).strip() if m else p.stem.replace('_',' ').title()
        body=re.sub(r'^#\s+.+$','',raw,count=1,flags=re.M).strip()
        chunks += [Chunk(p.name,title,t,i) for i,t in enumerate(chunk_text(body))]
    return chunks
CHUNKS=load_docs(); VECTORIZER=TfidfVectorizer(lowercase=True,stop_words='english',ngram_range=(1,2),sublinear_tf=True)
MATRIX=VECTORIZER.fit_transform([c.text for c in CHUNKS]) if CHUNKS else None

def retrieve(question,top_k=TOP_K):
    if MATRIX is None:return []
    scores=cosine_similarity(VECTORIZER.transform([question]),MATRIX).ravel()
    ranked=sorted(enumerate(scores),key=lambda x:x[1],reverse=True)[:top_k]
    return [{'source':CHUNKS[i].source,'title':CHUNKS[i].title,'chunk_id':CHUNKS[i].index,'text':CHUNKS[i].text,'score':round(float(s),4)} for i,s in ranked if s>0]

def call_llm(question,evidence):
    key=os.getenv('OPENAI_API_KEY')
    if not key or not evidence:return None
    url=os.getenv('OPENAI_BASE_URL','https://api.openai.com/v1').rstrip('/')+'/chat/completions'
    context='\n\n'.join(f"[{e['source']}] {e['text']}" for e in evidence)
    payload={'model':os.getenv('OPENAI_MODEL','gpt-4o-mini'),'temperature':0.1,'messages':[
      {'role':'system','content':'You are an energy operations document assistant. Answer only from supplied evidence. Do not add outside facts or invent procedures. If evidence is insufficient, say so. Cite operational claims using [source filename].'},
      {'role':'user','content':f'Question: {question}\n\nEvidence:\n{context}\n\nAnswer concisely with citations.'}]}
    req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={'Authorization':f'Bearer {key}','Content-Type':'application/json'},method='POST')
    try:
        with urllib.request.urlopen(req,timeout=25) as r: data=json.loads(r.read().decode())
        answer=data['choices'][0]['message']['content'].strip()
        # Reject responses that omit citations or cite documents outside retrieved evidence.
        cited=set(re.findall(r'\[([^\]]+\.md)\]',answer))
        allowed={e['source'] for e in evidence}
        return answer if cited and cited.issubset(allowed) else None
    except (urllib.error.URLError,TimeoutError,KeyError,IndexError,json.JSONDecodeError):return None

def fallback_answer(evidence):
    if not evidence or evidence[0]['score']<MIN_RELEVANCE:
        return 'I could not find sufficiently relevant information in the supplied documents. Please consult the applicable approved SOP or an authorized operations/safety professional.'
    return '\n\n'.join(f"{e['text']} [{e['source']}]" for e in evidence if e['score']>=MIN_RELEVANCE)

@app.get('/api/health')
def health():
    return {'status':'ok','documents':len({c.source for c in CHUNKS}),'chunks':len(CHUNKS),'retrieval_method':'TF-IDF vectors + cosine similarity','llm_enabled':bool(os.getenv('OPENAI_API_KEY'))}
@app.get('/api/demo')
def demo():
    return {'case_study':'Energy Operations Document RAG Assistant','status':'ready','documents_loaded':sorted({c.source for c in CHUNKS}),'chunks_indexed':len(CHUNKS),'retrieval':'TF-IDF vectorization with cosine similarity','generation':'LLM when configured; grounded extractive fallback otherwise'}
@app.post('/api/run')
def run_case(query:Query):
    evidence=retrieve(query.text,query.top_k); llm=call_llm(query.text,evidence)
    abstained=not evidence or evidence[0]['score']<MIN_RELEVANCE
    return {'question':query.text,'answer':llm or fallback_answer(evidence),'abstained':abstained,
      'generation_mode':'llm' if llm else 'grounded_extractive_fallback','sources':list(dict.fromkeys(e['source'] for e in evidence if e['score']>=MIN_RELEVANCE)),
      'retrieval_evidence':evidence,'safety_note':'Verify operational and safety-critical decisions against the current approved SOP and authorized personnel.'}
@app.get('/api/evaluate')
def evaluate():
    p=DATA/'questions.csv'; rows=list(csv.DictReader(p.open(encoding='utf-8'))) if p.exists() else []; results=[]; hits=0
    for row in rows:
        found=retrieve(row['question'],1); source=found[0]['source'] if found else None; hit=source==row['expected_source']; hits+=int(hit)
        results.append({'question':row['question'],'expected_source':row['expected_source'],'retrieved_source':source,'top1_hit':hit,'score':found[0]['score'] if found else 0})
    return {'test_cases':len(rows),'top1_source_accuracy':round(hits/len(rows),4) if rows else None,'results':results,'note':'Tiny retrieval smoke test only; not a statistically reliable quality estimate.'}
if FRONTEND.exists():app.mount('/',StaticFiles(directory=FRONTEND,html=True),name='frontend')

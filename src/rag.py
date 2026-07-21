from retrieval import retrieve
from utils import format_docs
from reranker import rerank

question = 'Who is the child according to the constitution'

docs = retrieve(question)
reranked = rerank(question, docs)
context = format_docs(reranked)

print('-----------')
print(context[:1000])

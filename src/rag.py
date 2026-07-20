from retrieval import retrieve
from utils import format_docs
from reranker import rerank

question = 'How can I become a citizen of Ghana'

docs = retrieve(question)
reranked = rerank(question, docs)
context = format_docs(reranked)

print('-----------')
print(context[:1000])

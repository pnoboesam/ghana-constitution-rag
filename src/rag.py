from retrieval import retrieve
from utils import format_docs

question = 'How can I become a citizen of Ghana'

docs = retrieve(question)
context = format_docs(docs)

print(len(docs))
print('-----------')
print(context[:1000])

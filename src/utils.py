def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        doc_content = ""
        router = doc.metadata['type']

        if router == 'front_matter':
            doc_content = f"Page: {doc.metadata['page_label']}\n\nContent:\n{doc.page_content}"

        elif router == 'article':
            doc_content = f"Chapter: {doc.metadata['chapter']}\nArticle: {doc.metadata['article']}\nPage: {doc.metadata['page_start']}\n\nContent:\n{doc.page_content}"

        elif router == 'schedule':
            doc_content = f"Part: {doc.metadata['part']}\nSection: {doc.metadata['section']}\nPage: {doc.metadata['page_start']}\n\nContent:\n{doc.page_content}"

        elif router == 'oath':
            doc_content = f"Oath Title: {doc.metadata['title']}\nPage: {doc.metadata['page_start']}\n\nContent:\n{doc.page_content}"

        formatted_docs.append(doc_content)
        
    return "\n\n\n-------\n\n\n".join(formatted_docs)
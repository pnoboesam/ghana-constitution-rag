from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from config import BASE_DIR
import re

path = BASE_DIR / "data" / "ghanaian_constitution1992.pdf"

def parse_constitution(path):

    loader = PyPDFLoader(path)
    pages = loader.load()

    documents = []

    front_docs = pages[:23]

    for doc in front_docs:
        doc.metadata["type"] = "front_matter"

    documents.extend(front_docs)

    pages = pages[23:]


    # -----------------------------
    # Current parser state
    # -----------------------------


    mode = "constitution"

    current_chapter = "CHAPTER ONE"
    current_chapter_title = "THE CONSTITUTION"

    current_schedule = None
    current_part = None

    current_article = 1
    current_title = None

    current_text = []

    page_start = 24


    # -----------------------------
    # Regex patterns
    # -----------------------------
    chapter_pattern = re.compile(r"^CHAPTER\s+[A-Z\s-]+$")
    article_pattern = re.compile(r"^(\d+)\.$")
    part_pattern = re.compile(r"^PART\s+[IVXLC]+")
    first_schedule_pattern = re.compile(r"^FIRST SCHEDULE$")
    second_schedule_pattern = re.compile(r"^SECOND SCHEDULE$")
    oath_pattern = re.compile(r"^THE\s+.*OATH.*$")

    # -----------------------------
    # Helper function
    # -----------------------------

    def save_current_document(page_end):

        nonlocal current_text
        nonlocal current_article
        nonlocal current_title

        if not current_text:
            return

        text = "\n".join(current_text).strip()

        metadata = {
            "source": path.name,
            "page_start": page_start,
            "page_end": page_end,
        }

        if mode == "constitution":
            metadata["chapter"] = current_chapter
            metadata["chapter_title"] = current_chapter_title
            metadata["type"] = "article"
            metadata["article"] = current_article

        elif mode == "first_schedule":
            metadata["type"] = "schedule"
            metadata["schedule"] = current_schedule
            metadata["part"] = current_part
            metadata["section"] = current_article

        elif mode == "second_schedule":
            metadata["type"] = "oath"
            metadata["schedule"] = current_schedule
            metadata["title"] = current_title

        documents.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

        current_text = []
        current_title = current_article = None


    # -----------------------------
    # Main parser
    # -----------------------------
    for page_number, page in enumerate(pages, start=24):

        lines = page.page_content.splitlines()

        for i, line in enumerate(lines):

            line = line.strip()

            if not line:
                continue

            if re.match(r"^The\s+Constitution$", line):
                continue

            # -----------------------------------
            # FIRST SCHEDULE
            # -----------------------------------
            if first_schedule_pattern.match(line):

                save_current_document(page_number)

                mode = "first_schedule"
                current_schedule = "FIRST SCHEDULE"
                current_article = None
                current_part = None
                current_text = []

                continue

            # -----------------------------------
            # SECOND SCHEDULE
            # -----------------------------------
            if second_schedule_pattern.match(line):

                save_current_document(page_number)

                mode = "second_schedule"
                current_schedule = "SECOND SCHEDULE"
                current_title = None
                current_text = []

                continue

            # -----------------------------------
            # CHAPTER
            # -----------------------------------

            if mode == "constitution" and chapter_pattern.match(line):

                save_current_document(page_number)
                
                current_chapter = line

                # -----------------------------------
                # Chapter title
                # -----------------------------------

                j = i + 1

                while j < len(lines):
                    title = lines[j].strip()

                    if title:
                        current_chapter_title = title
                        break

                    j += 1

                continue

            if mode == "constitution" and line == current_chapter_title:
                continue

            
        
            # -----------------------------------
            # PART
            # -----------------------------------
            if mode == "first_schedule":

                if part_pattern.match(line):

                    current_part = line
                    continue

            # -----------------------------------
            # Article / Section
            # -----------------------------------
            article_match = article_pattern.match(line)

            if article_match and mode in ("constitution", "first_schedule"):

                save_current_document(page_number)

                current_article = article_match.group(1)

                page_start = page_number

                current_text = []

                continue

            # -----------------------------------
            # OATH TITLE
            # -----------------------------------

            if mode == "second_schedule":

                if oath_pattern.match(line):

                    save_current_document(page_number)

                    current_title = line

                    page_start = page_number

                    current_text = []

                    continue

            # -----------------------------------
            # Normal content
            # -----------------------------------
            current_text.append(line)

    # Save last document
    page_start = 215
    save_current_document(len(pages)+23)

    return documents

def clean_documents(documents):
    documents = documents
    for i, doc in enumerate(documents[23:299], start=23):
        if doc.metadata["chapter_title"] == "COMMISSION ON HUMAN RIGHTS":
            documents[i].metadata["chapter_title"] = "COMMISSION ON HUMAN RIGHTS AND ADMINISTRATIVE JUSTICE"

    return documents

def get_documents():
    documents = parse_constitution(path)
    documents = clean_documents(documents)

    documents = [ doc for doc in documents if len(doc.page_content) > 30 ]
    print(f"number of documents after parsing: {len(documents)}")
    
    return documents

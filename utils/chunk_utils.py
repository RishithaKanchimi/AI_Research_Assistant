from langchain.text_splitter import CharacterTextSplitter

# def get_text_chunks(text):
#     splitter = CharacterTextSplitter(separator="\n", chunk_size=20000, chunk_overlap=500)
#     return splitter.split_text(text)


# def get_text_chunks(text):
#     total_chars = len(text)

#     # Adjust based on size of input
#     if total_chars < 5000:
#         chunk_size = 1000
#     elif total_chars < 20000:
#         chunk_size = 2000
#     elif total_chars < 50000:
#         chunk_size = 3000
#     elif total_chars < 100000:
#         chunk_size = 4000
#     else:
#         chunk_size = 5000  # Don't go too large — most models can't handle >8000 tokens
#     chunk_overlap = int(0.1 * chunk_size)  # 10% overlap

#     splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     chunks = splitter.split_text(text)

#     print(f"[INFO] Using chunk_size: {chunk_size}, overlap: {chunk_overlap}, total chunks: {len(chunks)}")
#     return chunks



#.....................based on rapidOCR ......updating chunks code..................

import re
from langchain.text_splitter import CharacterTextSplitter

def clean_ocr_text(text):
    """
    Cleans OCR text:
    - Preserves <img> and table-like formatting.
    - Removes broken or overly redundant lines.
    """

    # Remove broken markdown image tags (optional: keep <img> HTML tags)
    text = re.sub(r"!\[.*?\]\(.*?\)", "[IMAGE]", text)  # replace with [IMAGE] placeholder

    # Preserve newlines for tables but collapse overly long gaps
    text = re.sub(r"\n{3,}", "\n\n", text)  # keep paragraph/table separation

    # Normalize spacing but preserve indentation for tables
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        # Keep leading spaces (for indentation/tables), normalize in-line spacing
        line = re.sub(r"[ \t]{2,}", "  ", line)  # double spaces allowed in tables
        line = re.sub(r"[ \t]+$", "", line)      # remove trailing space
        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)
    return cleaned_text.strip()


def get_text_chunks(text):
    """
    Splits cleaned OCR text into manageable chunks, preserving structure.
    """

    cleaned_text = clean_ocr_text(text)
    total_chars = len(cleaned_text)

    # Heuristics for OCR-heavy and large text
    if total_chars < 5000:
        chunk_size = 1000
    elif total_chars < 20000:
        chunk_size = 1500
    elif total_chars < 50000:
        chunk_size = 2500
    elif total_chars < 100000:
        chunk_size = 3000
    else:
        chunk_size = 4000  # upper cap for most models

    chunk_overlap = int(0.15 * chunk_size)  # 15% overlap for context continuity

    splitter = CharacterTextSplitter(
        separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(cleaned_text)

    print(f"[INFO] Using chunk_size: {chunk_size}, overlap: {chunk_overlap}, total chunks: {len(chunks)}")
    return chunks

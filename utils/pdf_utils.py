import os
import pymupdf4llm

# EXTRACTED_DIR = "extracted_files"
# os.makedirs(EXTRACTED_DIR, exist_ok=True)


#.................1.................
# def extract_text_from_pdfs(pdf_docs):
#     all_text = ""

#     for pdf in pdf_docs:
#         filename = pdf.filename.replace(" ", "_")
#         temp_path = os.path.join(EXTRACTED_DIR, filename)

#         with open(temp_path, "wb") as f:
#             f.write(pdf.file.read())

#         reader = pymupdf4llm.to_markdown(temp_path)
#         all_text += reader

#         output_file_path = os.path.join(EXTRACTED_DIR, f"{os.path.splitext(filename)[0]}.txt")
#         with open(output_file_path, "w", encoding="utf-8", errors="ignore") as out:
#             out.write(reader)

#     return all_text

#.....................2..................
# def extract_text_from_pdfs(pdf_docs):
#     all_text = ""
#     first_pdf_filename = None  # Capture base filename from first PDF

#     for index, pdf in enumerate(pdf_docs):
#         filename = pdf.filename.replace(" ", "_")
#         temp_path = os.path.join(EXTRACTED_DIR, filename)

#         with open(temp_path, "wb") as f:
#             f.write(pdf.file.read())

#         reader = pymupdf4llm.to_markdown(temp_path)
#         all_text += reader

#         output_file_path = os.path.join(EXTRACTED_DIR, f"{os.path.splitext(filename)[0]}.txt")
#         with open(output_file_path, "w", encoding="utf-8", errors="ignore") as out:
#             out.write(reader)

#         if index == 0:
#             # Use only the first filename as embedding base
#             first_pdf_filename = os.path.splitext(filename)[0]

#     return all_text, first_pdf_filename



#..............3................27-07-2025....
#.... to  get tables and images.......working.................
# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain_community.document_loaders.parsers import RapidOCRBlobParser
# import os

# def extract_text_from_pdfs(pdf_docs):
#     all_text = ""
#     first_pdf_filename = None

#     for index, pdf in enumerate(pdf_docs):
#         filename = pdf.filename.replace(" ", "_")
#         temp_path = os.path.join(EXTRACTED_DIR, filename)

#         with open(temp_path, "wb") as f:
#             f.write(pdf.file.read())

#         # Use LangChain's PyMuPDFLoader with image and table support
#         loader = PyMuPDFLoader(
#             temp_path,
#             mode="page",
#             images_inner_format="markdown-img",   # embeds image as markdown
#             images_parser=RapidOCRBlobParser()    # parses embedded images via OCR
#         )

#         docs = loader.load()
#         document_text = "\n\n".join([doc.page_content for doc in docs])
#         all_text += document_text

#         # Save as text file (optional)
#         output_file_path = os.path.join(EXTRACTED_DIR, f"{os.path.splitext(filename)[0]}.txt")
#         with open(output_file_path, "w", encoding="utf-8", errors="ignore") as out:
#             out.write(document_text)

#         if index == 0:
#             first_pdf_filename = os.path.splitext(filename)[0]

#     return all_text, first_pdf_filename



#############................4..........................
# import os
# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain_community.document_loaders.parsers import TesseractBlobParser

# EXTRACTED_DIR = "./extracted_pdfs"

# def extract_text_from_pdfs(pdf_docs):
#     from uuid import uuid4

#     os.makedirs(EXTRACTED_DIR, exist_ok=True)
#     all_text = ""
#     first_pdf_filename = None

#     for index, pdf in enumerate(pdf_docs):
#         filename = pdf.filename.replace(" ", "_")
#         temp_path = os.path.join(EXTRACTED_DIR, filename)

#         # Skip if already extracted
#         output_file_path = os.path.join(EXTRACTED_DIR, f"{os.path.splitext(filename)[0]}.txt")
#         if os.path.exists(output_file_path):
#             print(f"[INFO] Skipping {filename}, already extracted.")
#             if index == 0:
#                 first_pdf_filename = os.path.splitext(filename)[0]
#             with open(output_file_path, "r", encoding="utf-8", errors="ignore") as f:
#                 all_text += f.read()
#             continue

#         # Save uploaded file to disk
#         with open(temp_path, "wb") as f:
#             f.write(pdf.file.read())

#         print(f"[INFO] Extracting from {filename} using Tesseract OCR...")
#         loader = PyMuPDFLoader(
#             temp_path,
#             mode="page",
#             images_inner_format="html-img",  # Better for table/image layout
#             images_parser=TesseractBlobParser()
#         )
#         docs = loader.load()
#         combined_text = "\n\n".join([doc.page_content for doc in docs])

#         # Save extracted content to a .txt file
#         with open(output_file_path, "w", encoding="utf-8", errors="ignore") as out:
#             out.write(combined_text)

#         all_text += combined_text

#         if index == 0:
#             first_pdf_filename = os.path.splitext(filename)[0]

#     return all_text, first_pdf_filename



import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.parsers import TesseractBlobParser

EXTRACTED_DIR = "./extracted_texts"  # ensure this folder exists

def extract_text_from_pdfs(pdf_docs):
    all_text = ""
    first_pdf_filename = None

    os.makedirs(EXTRACTED_DIR, exist_ok=True)

    for index, pdf in enumerate(pdf_docs):
        filename = pdf.filename.replace(" ", "_")
        temp_path = os.path.join(EXTRACTED_DIR, filename)

        # Save uploaded file
        with open(temp_path, "wb") as f:
            f.write(pdf.file.read())

        # Use LangChain loader with Tesseract for OCR and image+table extraction
        loader = PyMuPDFLoader(
            temp_path,
            mode="page",
            images_inner_format="html-img",
            images_parser=TesseractBlobParser()
        )

        try:
            docs = loader.load()
            page_texts = [doc.page_content for doc in docs]
            pdf_text = "\n\n".join(page_texts)
        except Exception as e:
            print(f"[ERROR] Failed to parse PDF using PyMuPDFLoader: {filename}: {e}")
            pdf_text = ""  # fallback

        if not pdf_text.strip():
            # fallback using pymupdf4llm markdown method
            try:
                from pymupdf4llm import to_markdown
                pdf_text = to_markdown(temp_path)
                print(f"[INFO] Used fallback markdown extraction for {filename}")
            except Exception as fallback_error:
                print(f"[ERROR] Fallback extraction failed for {filename}: {fallback_error}")
                pdf_text = ""

        all_text += pdf_text + "\n\n"

        # Save extracted content to .txt
        output_file_path = os.path.join(EXTRACTED_DIR, f"{os.path.splitext(filename)[0]}.txt")
        with open(output_file_path, "w", encoding="utf-8", errors="ignore") as out:
            out.write(pdf_text)

        if index == 0:
            first_pdf_filename = os.path.splitext(filename)[0]

    return all_text.strip(), first_pdf_filename

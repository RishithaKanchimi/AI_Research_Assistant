import uuid ,os
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from typing import List



##########################################################################
llm = AzureChatOpenAI(
    azure_endpoint="https://demoprep.openai.azure.com",
    api_key="BUGu6buSlFdXQITe38BF8njvsWqc6VTaYXlmAzTKpxLceZIOkhQKJQQJ99BGACYeBjFXJ3w3AAABACOGaR7X",
    azure_deployment="gpt-4o-mini",
    api_version="2024-10-21",
    temperature=0,
)


embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-large",
    azure_endpoint="https://demoprep.openai.azure.com",
    api_key="BUGu6buSlFdXQITe38BF8njvsWqc6VTaYXlmAzTKpxLceZIOkhQKJQQJ99BGACYeBjFXJ3w3AAABACOGaR7X",
    api_version="2024-10-21",
)



# Custom prompt
custom_prompt_template = """You are a financial assistant...

Context:
{context}

Question:
{question}
"""

custom_prompt = PromptTemplate(
    template=custom_prompt_template, input_variables=["context", "question"]
)
#...................1...................

# def get_vectorstore(text_chunks):
#     metadatas = [{"text": chunk[:200]} for chunk in text_chunks]
#     return Qdrant.from_texts(
#         texts=text_chunks,
#         embedding=embeddings,
#         metadatas=metadatas,
#         path=f"./db_{uuid.uuid4().hex}",
#         collection_name="document_embeddings1",
#     )
############################################
#...............2......................

def get_vectorstore(text_chunks, filename):
    metadatas = [{"text": chunk[:200]} for chunk in text_chunks]

    embedding_folder = "Embedding_Data"
    os.makedirs(embedding_folder, exist_ok=True)

    base_name = os.path.splitext(filename)[0].replace(" ", "_")
    db_path = os.path.join(embedding_folder, f"db_{base_name}")

    return Qdrant.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        metadatas=metadatas,
        path=db_path,
        collection_name="document_embeddings1",
    )




class CustomHybridRetriever(BaseRetriever):
    vectorstore: Qdrant
    k: int = 5
    alpha: float = 0.5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        docs = self.vectorstore.similarity_search(query, k=self.k)
        scored = []
        for doc in docs:
            text = doc.page_content
            keyword_overlap = sum(
                1 for term in query.lower().split() if term in text.lower()
            )
            hybrid_score = self.alpha + (1 - self.alpha) * keyword_overlap
            scored.append((hybrid_score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored]


def get_qa_chain(vectorstore):
    retriever = CustomHybridRetriever(vectorstore=vectorstore, k=5, alpha=0.5)
    return RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, chain_type_kwargs={"prompt": custom_prompt}
    )

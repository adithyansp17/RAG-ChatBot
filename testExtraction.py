from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, UnstructuredHTMLLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def upload_htmls():
    print("Uploading HTMLs...")
    loader = DirectoryLoader(
        path="hr-policies",
        glob="**/*.html",
        loader_cls=UnstructuredHTMLLoader
    )
    
    documents = loader.load()
    if len(documents) == 0:
        print("No documents found in the directory.")
        return
    
    print(f"Loaded {len(documents)} documents.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""])
    split_documents = text_splitter.split_documents(documents=documents)
    print(f"Split into {len(split_documents)} documents.")
    
    embeddings = OpenAIEmbeddings(openai_api_key="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")
    db = FAISS.from_documents(split_documents, embeddings)
    db.save_local("faiss_index")
    print("FAISS index saved.")


def faiss_query():
    print("Loading FAISS index...")
    embeddings = OpenAIEmbeddings(openai_api_key="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")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    print("FAISS index loaded.")

    query = "example query"  
    docs = new_db.similarity_search(query)
    print(f"Found {len(docs)} documents matching the query.")

    for doc in docs:
        print('here')
        print(doc)

if __name__ == "__main__":
    upload_htmls()
    faiss_query()
 

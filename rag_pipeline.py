import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

def build_vectorstore():
    print("Loading documents...")
    loader = DirectoryLoader(DOCS_DIR, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("Creating embeddings and storing in ChromaDB...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print("Vectorstore created successfully!")
    return vectorstore

def load_vectorstore():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    return vectorstore

def get_chain(vectorstore):
    llm = OllamaLLM(model="llama3.2", temperature=0.1)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate.from_template("""You are a DevOps assistant. Use the following context to answer the question.
If the answer is not in the context, say "I don't have information about that in my knowledge base."
Keep your answer clear and concise.

Context:
{context}

Question: {question}

Answer:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

def answer_question(question, chain, retriever):
    answer = chain.invoke(question)
    docs = retriever.invoke(question)
    sources = list(set([
        os.path.basename(doc.metadata["source"])
        for doc in docs
    ]))
    return answer, sources

if __name__ == "__main__":
    if not os.path.exists(CHROMA_DIR):
        vectorstore = build_vectorstore()
    else:
        print("Loading existing vectorstore...")
        vectorstore = load_vectorstore()

    chain, retriever = get_chain(vectorstore)

    print("\nDevOps RAG Assistant ready! Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue
        print("\nThinking...")
        answer, sources = answer_question(question, chain, retriever)
        print(f"\nAssistant: {answer}")
        print(f"Sources: {', '.join(sources)}\n")

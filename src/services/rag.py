import os
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import bs4
from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoaderfrom 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
import PyPDFLoader

def make_agent(): # Rate limiter
    # Getting LangChain to log everything
    langsmith = os.getenv("LANGSMITH_TRACING")

    # Getting API key
    huggingface = os.getenv("HUGGINGFACE_API_KEY")

    model = init_chat_model(
        "microsoft/Phi-3-mini-4k-instruct",
        model_provider="huggingface",
        temperature=0.7,
        max_tokens=1023,
        api_key=huggingface
    )

    # Initializing embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Storing the embeddings
    storage = InMemoryVectorStore(embeddings)

    # Loading in a source with consumer information for clothes
    # Background information to store for the LLM
    bs4_soup = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))

    fashion_loader = PyPDFLoader(
        "https://globusjournal.com/wp-content/uploads/2024/11/GMIT-JD24-161-7-Vibha-Chandrakar.pdf"
    )
    docs = fashion_loader.load()

    assert len(docs) == 1

    # Splitting docs into chunks
    chunk_maker = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=10,
        add_start_index=True, #track index in original document
    )

    total_chunks = chunk_maker.split_documents(docs)

    # Embed and store all the chunks of documents
    documents = storage.add_documents(documents=total_chunks)

    @tool(response_format="context_and_artifact")
    def retrieval_context(query: str):
        '''retrieve information to help answer a query'''
        documents = make_agent()
        retrieved_docs = documents.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (f"source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    tools = [retrieval_context]

    prompt = (
        "You are a data scientist that is given information regarding"
        "an employers clothing idea. Based on specific criteria, give"
        "predictions on how that piece of clothing will do in the market"
        "I will give predictions on profit margin, quantity sold, and total"
        "amount of items sold. You must take this information along with the"
        "sources given and generate a summary and predictions."
    )
    agent = create_agent(model, tools, system_prompt=prompt)

    return agent

async def get_rag_response(query: str):
    agent = make_agent()
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return response
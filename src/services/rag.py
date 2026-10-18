import os
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2] # Project root, so paths work from any cwd
PDF_PATH = BASE_DIR / "data/context/fashion_info.pdf"
NAMESPACE = "fashion-research"

SYSTEM_PROMPT = """
The user uploaded a picture of clothing and wants to find similar clothes.
You are given the color, category, and attributes the image model predicted.
Always call the retrieval_context tool first to look up recent fashion trends
related to the clothing. Then describe the clothing they uploaded using only the
predicted color, category, and attributes (you cannot see the image, so don't
invent details like zippers, collars, or fit). Explain how
it fits with the trends you found, and suggest similar styles to look for.
Only use trend information from the retrieved context. If nothing relevant
is found, say so instead of making trends up.
"""

def get_embeddings():
    # Groq has no embedding models, so Pinecone's hosted model makes the vectors.
    # llama-text-embed-v2 outputs 1024 numbers, which must match the index dimension
    return PineconeEmbeddings(
        model="llama-text-embed-v2",
        pinecone_api_key=os.environ["PINECONE_API_KEY"],
    )

@lru_cache(maxsize=1)
def get_vector_store():
    """Connect to an existing Pinecone index. Cached so it is only built once per process"""
    # Made here instead of at import so the app can still start without the keys
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

    return PineconeVectorStore(
        index=index, # index_name only takes a string, the Index object goes here
        embedding=get_embeddings(),
        namespace=NAMESPACE,
    )


def ingest_pdf():
    """Run once to load fashion_info.pdf and index its chunks."""
    # Only needed for ingesting, so the api doesn't have to install these
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if not PDF_PATH.is_file():
        raise FileNotFoundError(
            f"Could not find {PDF_PATH}"
        )

    loader = PyPDFLoader(str(PDF_PATH))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)

    # Attach source metadata for traceability.
    for chunk in chunks:
        chunk.metadata["source"] = PDF_PATH.name

    store = get_vector_store()
    store.add_documents(chunks)

    return len(chunks)


@tool(response_format="content_and_artifact")
def retrieval_context(query: str):
    """Search fashion research in Pinecone for relevant evidence."""
    docs = get_vector_store().similarity_search(query, k=4)

    if not docs:
        return "No relevant documents found.", []

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}, "
        f"page: {doc.metadata.get('page', 'unknown')}\n"
        f"{doc.page_content}"
        for doc in docs
    )

    return context, docs


@lru_cache(maxsize=1)
def make_agent():
    """Create the Groq agent and connect its retrieval tool. Built once and reused"""
    model = ChatGroq(
        model="openai/gpt-oss-120b", # Supports tool calling on Groq
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0.2,
        max_tokens=2048,
    )

    return create_agent(
        model,
        tools=[retrieval_context],
        system_prompt=SYSTEM_PROMPT,
    )

def build_query(prediction: dict):
    """Turns the {color, category, attributes} dict from services/predict.py into a question"""
    attrs = ", ".join(prediction.get("attributes") or []) or "none detected"
    return (
        f"I uploaded a {prediction['color']} {prediction['category']}. "
        f"Attributes: {attrs}. "
        "Describe it and tell me how it fits with recent fashion trends."
    )

async def get_rag_response(query: str):
    agent = make_agent()

    response = await agent.ainvoke({
        "messages": [
            {"role": "user", "content": query}
        ]
    })

    # Return the final assistant message, not the entire state.
    for message in reversed(response["messages"]):
        if getattr(message, "type", None) == "ai":
            # .text joins the content into a string even when it comes back as a list of blocks
            if message.text:
                return message.text

    return "The agent did not return a response."

if __name__ == '__main__':
    # Run this once to upsert the documents into Pinecone: python -m src.services.rag
    chunks = ingest_pdf()
    print(f"Indexed {chunks} chunks into namespace '{NAMESPACE}'")

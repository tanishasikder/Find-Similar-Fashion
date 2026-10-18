
import os
from pathlib import Path
from pinecone import Pinecone
from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PDF_PATH = Path("fashion.pdf")
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
NAMESPACE = "fashion-research"

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=os.environ["GOOGLE_API_KEY"],
        output_dimensionality=768,
    )

def get_vector_store():
    """Connect to an existing Pinecone index."""
    index = pc.Index(
        os.environ["PINECONE_INDEX_NAME"]
    )
    return PineconeVectorStore(
        index_name=index,
        embedding=get_embeddings(),
        namespace=NAMESPACE,
    )


def ingest_pdf():
    """Run once to load fashion.pdf and index its chunks."""
    if not PDF_PATH.is_file():
        raise FileNotFoundError(
            f"Could not find {PDF_PATH.resolve()}"
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


def make_agent():
    """Create the Gemini agent and connect its retrieval tool."""
    model = init_chat_model(
        "google_genai:gemini-2.5-flash",
        temperature=0.2,
        max_tokens=2048,
    )

    store = get_vector_store()

    from langchain.tools import tool

    @tool(response_format="content_and_artifact")
    def retrieval_context(query: str):
        """Search fashion research in Pinecone for relevant evidence."""
        docs = store.similarity_search(query, k=4)

        if not docs:
            return "No relevant documents found.", []

        context = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'unknown')}, "
            f"page: {doc.metadata.get('page', 'unknown')}\n"
            f"{doc.page_content}"
            for doc in docs
        )

        return context, docs

    system_prompt = """
        The user wants to find similar clothes to what they
        uploaded. Describe the clothes they uploaded along with
        information given about recent fashion trends/

        Context: {context}
    """

    from langchain.agents import create_agent

    return create_agent(
        model,
        tools=[retrieval_context],
        system_prompt=system_prompt,
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
            if message.content:
                return message.content

    return "The agent did not return a response."

if __name__ == '__main__':
    # Run this once to upsert the documents into Pinecone
    vector_store = get_vector_store()
    chunks = ingest_pdf()
    vector_store.add_documents(documents=chunks)

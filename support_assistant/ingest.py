import os
from typing import TypedDict, Literal, Union
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
from langgraph.graph import StateGraph, START, END
from models import FinalAnswerSchema

# get the folder where this file lives so paths work correctly everywhere
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
docs_folder = os.path.join(BASE_DIR, "docs")
chroma_folder = os.path.join(BASE_DIR, "chroma_db")

# load the sentence transformer model for creating embeddings
# all-MiniLM-L6-v2 is a small and fast model that runs fully offline
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# set up the chromadb client to store embeddings on disk
chroma_client = chromadb.PersistentClient(path=chroma_folder)

# always start fresh so re-running does not create duplicate entries
try:
    chroma_client.delete_collection("support_documents")
except Exception:
    pass

doc_store = chroma_client.get_or_create_collection("support_documents")

# read each txt file from the docs folder, embed it and store in chromadb
doc_index = 0
for root, dirs, files in os.walk(docs_folder):
    for file_name in files:
        if file_name.endswith(".txt"):
            full_path = os.path.join(root, file_name)
            print(f"Ingesting: {full_path}")
            with open(full_path, "r", encoding="utf-8") as f:
                text_content = f.read()
                # convert the text into a vector embedding
                embedding_vector = embed_model.encode(text_content)
                doc_store.add(
                    embeddings=[embedding_vector.tolist()],
                    documents=[text_content],
                    metadatas=[{"source": file_name}],
                    ids=[str(doc_index + 1)]
                )
                doc_index += 1

print(f"Total documents ingested: {doc_index}")

# print all stored documents to confirm ingestion worked
for meta, doc in zip(
    doc_store.get(include=["metadatas"])["metadatas"],
    doc_store.get(include=["documents"])["documents"]
):
    print(f"Stored -> source: {meta}, preview: {doc[:80]}")


# -----------------------------------------------------------------------
# Task 2: Structured Prompt Template
# This prompt follows the role-context-task-format-length skeleton.
# It includes a negative constraint and two few-shot examples.
# This template is used only when MOCK_LLM=0 (the optional real LLM path).
# -----------------------------------------------------------------------
SUPPORT_PROMPT = """
ROLE: You are a Zepto customer support AI assistant. Your job is to help
customers with questions about delivery, returns, membership, order tracking,
cancellations, gift cards, and support hours. You only use information from
the documents provided below. You do not guess or make things up.

CONTEXT: Here are the retrieved support documents:
{retrieved_documents}

TASK: Answer the customer's question using only the information in the
documents above. Do not answer using information not present in the provided
context. If the documents do not have enough information to answer the
question, say: "I cannot answer this question using the available documents."

FORMAT: Give only the final answer. Do not mention document titles or IDs
unless the customer asks for sources.

LENGTH: Keep your answer short, ideally 2 to 4 sentences.

FEW-SHOT EXAMPLES:

User: Can I order a pizza from Zepto?
Response: I cannot answer this question using the available documents.

User: How long do I have to report a damaged item?
Response: You must report damaged or spoiled items within 24 hours of delivery
using the Report an Issue button on the order page. Zepto will send a free
replacement or give a full refund without needing you to return the item,
unless the order value is over INR 1000 in which case you need to submit a photo.

CUSTOMER QUESTION: {query}
"""


# -----------------------------------------------------------------------
# LangGraph State
# -----------------------------------------------------------------------
class GraphState(TypedDict):
    query: str
    intent: Literal["policy_question", "general_question"]
    retrieved_docs: list[str]
    answer: Union[dict, str]


# -----------------------------------------------------------------------
# Node 1: classify_intent
# Decides if the query needs policy retrieval or not
# -----------------------------------------------------------------------
def classify_intent(state: GraphState) -> dict:
    mock_llm = os.getenv("MOCK_LLM", "1")
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")

    if mock_llm == "0" and api_key:
        # optional MOCK_LLM=0 extension: call the real LLM to classify
        llm_client = Groq(api_key=api_key)
        classify_prompt = (
            f"Classify this customer query into exactly one category: "
            f"'policy_question' or 'general_question'.\n"
            f"Query: {state['query']}\n"
            f"Return only 'policy_question' or 'general_question'."
        )
        llm_response = llm_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": classify_prompt}]
        )
        response_text = llm_response.choices[0].message.content.strip().lower()
        intent = "policy_question" if "policy" in response_text else "general_question"
    else:
        # mock mode (graded baseline): use a keyword heuristic, no LLM call
        policy_keys = [
            "delivery", "return", "refund", "membership",
            "tracking", "cancel", "gift card", "support hours"
        ]
        query_lower = state["query"].lower()
        if any(keyword in query_lower for keyword in policy_keys):
            intent = "policy_question"
        else:
            intent = "general_question"

    print(f"[classify_intent] '{state['query']}' classified as: '{intent}'")
    return {"intent": intent}


# -----------------------------------------------------------------------
# Conditional edge: route to the right node based on intent
# -----------------------------------------------------------------------
def route_query(state: GraphState) -> Literal["retrieve_and_answer", "direct_answer"]:
    current_intent = state.get("intent")
    print(f"[route_query] Routing intent: '{current_intent}'")
    if current_intent == "policy_question":
        return "retrieve_and_answer"
    return "direct_answer"


# -----------------------------------------------------------------------
# Node 2: retrieve_and_answer
# Embeds the query, retrieves top 3 chunks, builds the answer
# -----------------------------------------------------------------------
def retrieve_and_answer(state: GraphState) -> dict:
    print("[retrieve_and_answer] Running vector search...")

    # embed the query and search chromadb - this always runs for real
    query_vector = embed_model.encode(state["query"])
    search_results = doc_store.query(
        query_embeddings=[query_vector.tolist()],
        n_results=3,
        include=["documents", "metadatas"]
    )

    retrieved_texts = search_results.get("documents", [[]])[0]
    retrieved_ids = search_results.get("ids", [[]])[0]
    retrieved_metas = search_results.get("metadatas", [[]])[0]

    # get the source filenames from metadata
    source_names = [
        meta.get("source", doc_id)
        for meta, doc_id in zip(retrieved_metas, retrieved_ids)
    ] if retrieved_metas else retrieved_ids

    mock_llm = os.getenv("MOCK_LLM", "1")
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")

    if mock_llm == "0" and api_key:
        # optional extension: call the real LLM with retrieved context
        answer_data = call_llm_with_retry(state["query"], retrieved_texts, source_names)
    else:
        # mock mode (graded baseline): build a canned answer from the top chunk
        top_chunk = retrieved_texts[0][:200] if retrieved_texts else "No documents found."
        canned_answer = FinalAnswerSchema(
            answer=f"Based on the retrieved context: {top_chunk}",
            sources=source_names,
            confidence=1.0
        )
        answer_data = canned_answer.model_dump()

    return {"retrieved_docs": retrieved_texts, "answer": answer_data}


# -----------------------------------------------------------------------
# Node 3: direct_answer
# Used for general questions that do not need policy retrieval
# -----------------------------------------------------------------------
def direct_answer(state: GraphState) -> dict:
    print("[direct_answer] Handling general question...")

    mock_llm = os.getenv("MOCK_LLM", "1")
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")

    if mock_llm == "0" and api_key:
        # optional extension: call the real LLM directly without retrieval
        answer_data = call_llm_direct_with_retry(state["query"])
    else:
        # mock mode (graded baseline): return a fixed canned string
        canned_answer = FinalAnswerSchema(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=1.0
        )
        answer_data = canned_answer.model_dump()

    return {"answer": answer_data}


# -----------------------------------------------------------------------
# Optional LLM helpers (only used when MOCK_LLM=0)
# Includes retry logic on validation failure
# -----------------------------------------------------------------------
def call_llm_with_retry(query: str, docs: list[str], source_ids: list[str]) -> dict:
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")
    if not api_key:
        return FinalAnswerSchema(
            answer="I cannot answer this question using the available documents.",
            sources=[],
            confidence=0.0
        ).model_dump()

    llm_client = Groq(api_key=api_key)
    docs_text = "\n".join(docs)
    full_prompt = (
        SUPPORT_PROMPT.format(retrieved_documents=docs_text, query=query)
        + "\n\nRespond in valid JSON with keys: answer (string), sources (list of strings), confidence (float 0 to 1)."
    )

    messages = [
        {"role": "system", "content": "You are a helpful Zepto support AI. Output valid JSON only."},
        {"role": "user", "content": full_prompt}
    ]

    last_error = ""
    raw_content = ""
    for attempt in range(3):
        try:
            llm_response = llm_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=messages,
                response_format={"type": "json_object"}
            )
            raw_content = llm_response.choices[0].message.content
            validated = FinalAnswerSchema.model_validate_json(raw_content)
            return validated.model_dump()
        except Exception as err:
            last_error = str(err)
            print(f"[call_llm_with_retry] Attempt {attempt + 1} failed: {last_error}")
            messages.append({"role": "assistant", "content": raw_content})
            messages.append({
                "role": "user",
                "content": (
                    f"Validation failed: {last_error}. "
                    "Please return valid JSON with keys answer (string), "
                    "sources (list of strings), and confidence (float 0 to 1)."
                )
            })

    return FinalAnswerSchema(
        answer=f"Error: could not get a valid response after retries. {last_error}",
        sources=[],
        confidence=0.0
    ).model_dump()


def call_llm_direct_with_retry(query: str) -> dict:
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")
    if not api_key:
        return FinalAnswerSchema(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=0.0
        ).model_dump()

    llm_client = Groq(api_key=api_key)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful support assistant. Output valid JSON with keys answer (str), sources (empty list), confidence (float)."
        },
        {
            "role": "user",
            "content": f"Query: {query}\nReturn valid JSON with keys answer, sources (empty list), and confidence."
        }
    ]

    last_error = ""
    raw_content = ""
    for attempt in range(3):
        try:
            llm_response = llm_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=messages,
                response_format={"type": "json_object"}
            )
            raw_content = llm_response.choices[0].message.content
            validated = FinalAnswerSchema.model_validate_json(raw_content)
            return validated.model_dump()
        except Exception as err:
            last_error = str(err)
            print(f"[call_llm_direct_with_retry] Attempt {attempt + 1} failed: {last_error}")
            messages.append({"role": "assistant", "content": raw_content})
            messages.append({
                "role": "user",
                "content": (
                    f"Validation failed: {last_error}. "
                    "Return valid JSON with keys answer (string), sources (empty list), and confidence (float)."
                )
            })

    return FinalAnswerSchema(
        answer=f"Error: could not get a valid response after retries. {last_error}",
        sources=[],
        confidence=0.0
    ).model_dump()


# -----------------------------------------------------------------------
# Build the LangGraph StateGraph
# -----------------------------------------------------------------------
flow_builder = StateGraph(GraphState)

flow_builder.add_node("classify_intent", classify_intent)
flow_builder.add_node("retrieve_and_answer", retrieve_and_answer)
flow_builder.add_node("direct_answer", direct_answer)

flow_builder.add_edge(START, "classify_intent")

# conditional edge routes to the right node based on the classified intent
flow_builder.add_conditional_edges(
    "classify_intent",
    route_query,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)

flow_builder.add_edge("retrieve_and_answer", END)
flow_builder.add_edge("direct_answer", END)

# compile the graph so it is ready to use
support_graph = flow_builder.compile()


# quick test when running this file directly
if __name__ == "__main__":
    test_state: GraphState = {
        "query": "How do I cancel my order?",
        "intent": "general_question",
        "retrieved_docs": [],
        "answer": ""
    }
    result = support_graph.invoke(test_state)
    print("\nResult:\n", result.get("answer"))

import os
from typing import Literal
from schemas import GraphState, FinalAnswerSchema
from embed_store import sentence_encoder, knowledge_base
from llm_utils import ask_llm_for_answer, ask_llm_direct


# -----------------------------------------------------------------------
# Node 1 - classify_intent
# Reads the query and decides if it is a policy question or general question
# -----------------------------------------------------------------------
def classify_intent(state: GraphState) -> dict:
    use_mock = os.getenv("MOCK_LLM", "1")
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")

    if use_mock == "0" and api_key:
        # optional real LLM mode - ask Groq to classify the query
        from groq import Groq
        groq_client = Groq(api_key=api_key)
        classification_prompt = (
            f"Classify this customer question into one of two categories: "
            f"'policy_question' or 'general_question'.\n"
            f"Question: {state['query']}\n"
            f"Output only the category name."
        )
        llm_resp = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": classification_prompt}]
        )
        llm_text = llm_resp.choices[0].message.content.strip().lower()
        intent = "policy_question" if "policy" in llm_text else "general_question"
    else:
        # mock mode (graded baseline) - keyword check, no LLM call needed
        zepto_topics = [
            "delivery", "return", "refund", "membership",
            "tracking", "cancel", "gift card", "support hours"
        ]
        question_lower = state["query"].lower()
        intent = (
            "policy_question"
            if any(topic in question_lower for topic in zepto_topics)
            else "general_question"
        )

    print(f"[classify_intent] '{state['query']}' -> '{intent}'")
    return {"intent": intent}


# -----------------------------------------------------------------------
# Conditional edge - decides which node runs after classify_intent
# -----------------------------------------------------------------------
def pick_next_step(state: GraphState) -> Literal["retrieve_and_answer", "direct_answer"]:
    chosen = state.get("intent")
    print(f"[pick_next_step] routing to: '{chosen}'")
    return "retrieve_and_answer" if chosen == "policy_question" else "direct_answer"


# -----------------------------------------------------------------------
# Node 2 - retrieve_and_answer
# Embeds the query, finds the top 3 matching docs, builds the answer
# -----------------------------------------------------------------------
def retrieve_and_answer(state: GraphState) -> dict:
    print("[retrieve_and_answer] searching vector store...")

    # embed the query the same way we embedded the documents at ingestion time
    search_vector = sentence_encoder.encode(state["query"])
    vector_results = knowledge_base.query(
        query_embeddings=[search_vector.tolist()],
        n_results=3,
        include=["documents", "metadatas"]
    )

    matched_texts = vector_results.get("documents", [[]])[0]
    matched_ids   = vector_results.get("ids",       [[]])[0]

    use_mock = os.getenv("MOCK_LLM", "1")
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")

    if use_mock == "0" and api_key:
        # real LLM mode - let Groq generate the answer from the retrieved context
        answer_data = ask_llm_for_answer(state["query"], matched_texts)
    else:
        # mock mode (graded baseline) - canned answer from the top retrieved chunk
        main_chunk = matched_texts[0][:200] if matched_texts else "No relevant documents found."
        default_response = FinalAnswerSchema(
            answer=f"Based on the retrieved context: {main_chunk}",
            sources=matched_ids,   # chromadb document IDs e.g. "1", "5", "2"
            confidence=1.0
        )
        answer_data = default_response.model_dump()

    return {"retrieved_docs": matched_texts, "answer": answer_data}


# -----------------------------------------------------------------------
# Node 3 - direct_answer
# Handles out-of-scope questions that do not match any Zepto policy topic
# -----------------------------------------------------------------------
def direct_answer(state: GraphState) -> dict:
    print("[direct_answer] question is out of scope...")

    use_mock = os.getenv("MOCK_LLM", "1")
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")

    if use_mock == "0" and api_key:
        # real LLM mode - let Groq answer without any retrieval context
        answer_data = ask_llm_direct(state["query"])
    else:
        # mock mode (graded baseline) - fixed polite response, no LLM call
        default_response = FinalAnswerSchema(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=1.0
        )
        answer_data = default_response.model_dump()

    return {"answer": answer_data}

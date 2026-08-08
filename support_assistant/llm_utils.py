import os
from groq import Groq
from schemas import FinalAnswerSchema

# prompt template with role, context, task, format, length and two few-shot examples
# this is only used when MOCK_LLM=0 (the optional real LLM extension)
ZEPTO_SUPPORT_PROMPT = """
ROLE: You are a Zepto customer support AI. Help customers with questions about
delivery, returns, membership, order tracking, cancellations, damaged items,
gift cards, and support hours. Use only the documents below to answer.
Do not answer using information not present in the provided context.

CONTEXT: Policy documents retrieved for this query:
{retrieved_documents}

TASK: Answer the customer question using only the text in the documents above.
If the documents do not have enough information, say exactly:
"I cannot answer this question using the available documents."

FORMAT: Give a direct answer. Do not mention document names unless the
customer specifically asks for sources.

LENGTH: Keep the answer to 2 to 4 sentences maximum.

FEW-SHOT EXAMPLES:

Customer: Can I order medicines from Zepto?
Response: I cannot answer this question using the available documents.

Customer: How long do I have to report a damaged item?
Response: You must report damaged or missing items within 24 hours of delivery
using the Report an Issue button on your order page. Zepto will send a free
replacement or issue a full refund without asking you to return the item,
unless the order total was above INR 1000 in which case a photo is required.

CUSTOMER QUESTION: {query}
"""


def ask_llm_for_answer(query: str, docs: list) -> dict:
    """
    Sends the retrieved documents and user query to the Groq LLM and returns
    a validated FinalAnswerSchema dict. Retries up to 3 times if validation fails.
    Only called when MOCK_LLM=0.
    """
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")
    if not api_key:
        return FinalAnswerSchema(
            answer="I cannot answer this question using the available documents.",
            sources=[],
            confidence=0.0
        ).model_dump()

    groq_client = Groq(api_key=api_key)
    context_text = "\n".join(docs)
    full_prompt = (
        ZEPTO_SUPPORT_PROMPT.format(retrieved_documents=context_text, query=query)
        + "\n\nRespond in valid JSON with exactly these keys: "
        "answer (string), sources (list of strings), confidence (float between 0 and 1)."
    )

    chat_messages = [
        {"role": "system", "content": "You are a Zepto support AI. Output valid JSON only."},
        {"role": "user", "content": full_prompt}
    ]

    last_err = ""
    raw_text = ""
    for attempt_num in range(3):
        try:
            chat_response = groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=chat_messages,
                response_format={"type": "json_object"}
            )
            raw_text = chat_response.choices[0].message.content
            validated = FinalAnswerSchema.model_validate_json(raw_text)
            return validated.model_dump()
        except Exception as err:
            last_err = str(err)
            print(f"[ask_llm_for_answer] attempt {attempt_num + 1} failed: {last_err}")
            chat_messages.append({"role": "assistant", "content": raw_text})
            chat_messages.append({
                "role": "user",
                "content": (
                    f"That response failed with: {last_err}. "
                    "Please return valid JSON with keys: "
                    "answer (string), sources (list of strings), confidence (float)."
                )
            })

    return FinalAnswerSchema(
        answer=f"Error: could not get a valid response after 3 tries. {last_err}",
        sources=[],
        confidence=0.0
    ).model_dump()


def ask_llm_direct(query: str) -> dict:
    """
    Sends a general question directly to the Groq LLM with no retrieval context.
    Retries up to 3 times on validation failure.
    Only called when MOCK_LLM=0.
    """
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")
    if not api_key:
        return FinalAnswerSchema(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=0.0
        ).model_dump()

    groq_client = Groq(api_key=api_key)
    chat_messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful Zepto customer support assistant. "
                "Return valid JSON with keys: answer (str), sources (empty list), confidence (float)."
            )
        },
        {
            "role": "user",
            "content": (
                f"Customer question: {query}\n"
                "Return JSON with keys answer, sources (empty list), and confidence."
            )
        }
    ]

    last_err = ""
    raw_text = ""
    for attempt_num in range(3):
        try:
            chat_response = groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=chat_messages,
                response_format={"type": "json_object"}
            )
            raw_text = chat_response.choices[0].message.content
            validated = FinalAnswerSchema.model_validate_json(raw_text)
            return validated.model_dump()
        except Exception as err:
            last_err = str(err)
            print(f"[ask_llm_direct] attempt {attempt_num + 1} failed: {last_err}")
            chat_messages.append({"role": "assistant", "content": raw_text})
            chat_messages.append({
                "role": "user",
                "content": (
                    f"Validation error: {last_err}. "
                    "Please return JSON with keys answer (string), sources (empty list), and confidence (float)."
                )
            })

    return FinalAnswerSchema(
        answer=f"Error: could not get a valid response after 3 tries. {last_err}",
        sources=[],
        confidence=0.0
    ).model_dump()

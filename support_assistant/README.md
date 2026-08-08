# Support Assistant - Zepto RAG Pipeline

This module builds a small but complete support assistant for Zepto
using retrieval-augmented generation (RAG). It embeds 8 Zepto policy
documents, routes each incoming question through a LangGraph pipeline,
retrieves the most relevant policy text from ChromaDB, and returns a
structured JSON answer.

The whole thing works offline using a mock mode for the LLM. No API key
or internet access is needed to run and grade this module.

---

## How to Run Locally

Install the required packages and start the server.

```
pip install -r requirements.txt
uvicorn app:zepto_app --host 0.0.0.0 --port 8000
```

To run with real LLM calls (optional, not graded):

```
MOCK_LLM=0 GROQ_API_KEY=your_key_here uvicorn app:zepto_app --host 0.0.0.0 --port 8000
```

---

## How to Build and Run with Docker

```
docker build -t support-assistant .
docker run -p 7860:7860 support-assistant
```

The Dockerfile uses python:3.10-slim which supports all required packages
including sentence-transformers and torch. The FastAPI instance is named
zepto_app so the CMD passes app:zepto_app to uvicorn.

---

## Recorded API Responses (MOCK_LLM at default)

Both examples below were captured with the server started normally
without setting MOCK_LLM, so mock mode is active.

### Example 1 - Policy Question (retrieval is triggered)

Request:
```
curl -s -X POST "http://127.0.0.1:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"query": "How to cancel my order?"}'
```

Raw JSON Response:
```
{
  "query": "How to cancel my order?",
  "intent": "policy_question",
  "retrieved_docs": [
    "Order Cancellation Policy: \"Orders can be cancelled free of cost any time before the order status changes to 'Packed', typically within the first 2 minutes of placing the order. Once an order has been packed, it can no longer be cancelled through the app, since the rider is dispatched immediately after packing given Zepto's quick-delivery model. If a packed order cannot be delivered due to a Zepto-side issue (for example, rider unavailability), the order is auto-cancelled and fully refunded without any cancellation fee.\"",
    "Damaged or Missing Items: \"If an order arrives with damaged, spoiled, or missing items, customers must report it within 24 hours of delivery through the 'Report an Issue' button on the order page. Zepto ships a free replacement or issues a full refund for damaged, spoiled, or missing items without requiring the customer to return the original item, unless the order value exceeds INR 1000, in which case a photo of the issue must be submitted before a replacement or refund is processed.\"",
    "Returns and Refunds: \"Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in unopened, resalable condition. Approved refunds are credited to the original payment method within 3-5 business days, or instantly to the Zepto wallet if the customer opts for wallet credit.\""
  ],
  "answer": {
    "answer": "Based on the retrieved context: Order Cancellation Policy: \"Orders can be cancelled free of cost any time before the order status changes to 'Packed', typically within the first 2 minutes of placing the order. Once an order has been",
    "sources": ["5", "6", "2"],
    "confidence": 1.0
  }
}
```

What happened: the word "cancel" matched a keyword in the zepto_topics list
so classify_intent set intent to policy_question. The retrieve_and_answer
node embedded the query, searched ChromaDB by cosine similarity, and got
the top 3 matching chunks. The mock answer was built from the first 200
characters of the top chunk. Sources are the ChromaDB document IDs.

---

### Example 2 - General Question (retrieval is NOT triggered)

Request:
```
curl -s -X POST "http://127.0.0.1:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the capital of France?"}'
```

Raw JSON Response:
```
{
  "query": "What is the capital of France?",
  "intent": "general_question",
  "retrieved_docs": [],
  "answer": {
    "answer": "I can only answer questions about Zepto policies right now.",
    "sources": [],
    "confidence": 1.0
  }
}
```

What happened: no keyword from zepto_topics matched so classify_intent
set intent to general_question. The direct_answer node returned the fixed
canned string without touching ChromaDB or making any LLM call.

---

## RAG Pipeline Architecture

The full pipeline has four stages. Data flows through each one in order.

### Stage 1 - Document Ingestion (embed_store.py)

When the FastAPI app starts, embed_store.py runs immediately on import.
It walks the docs folder and reads all 8 txt files (doc_01.txt to
doc_08.txt). Each file is treated as one document chunk. A metadata tag
with the source filename is attached so we can trace results back to
the original file.

This stage always runs the same way regardless of MOCK_LLM.

### Stage 2 - Embedding and Storage (embed_store.py)

Each document text is passed to sentence_encoder.encode() which uses the
all-MiniLM-L6-v2 model to produce a 384-dimensional float vector. The
vector, the original text, the metadata, and a numeric string ID are all
stored in a ChromaDB collection called zepto_policies saved in the
chroma_db folder on disk.

ChromaDB and the sentence-transformers model both run completely locally.
No API key and no internet connection are needed for this stage.

### Stage 3 - Intent Classification and Routing (nodes.py, graph.py)

When a query hits POST /ask, the LangGraph graph defined in graph.py
starts at the classify_intent node. This node decides between two
intents: policy_question or general_question.

This stage branches on MOCK_LLM.

Mock mode (MOCK_LLM unset or 1 - graded baseline):
classify_intent checks if any keyword from the zepto_topics list appears
in the lowercased query. Keywords are: delivery, return, refund,
membership, tracking, cancel, gift card, support hours. If any keyword
matches the intent is policy_question, otherwise general_question. No
LLM call is made.

Real LLM mode (MOCK_LLM=0 - optional extension):
classify_intent sends the query to llama3-8b-8192 on Groq and asks it
to return policy_question or general_question. The response is checked
for the word "policy" to set the intent.

After classification, pick_next_step is the conditional edge that reads
the intent and routes to retrieve_and_answer for policy questions or to
direct_answer for general questions. This routing logic does not depend
on MOCK_LLM.

### Stage 4 - Answer Generation (nodes.py, llm_utils.py)

For policy questions, retrieve_and_answer:
1. Encodes the query using sentence_encoder.encode() - always runs for real
2. Queries knowledge_base with n_results=3 to get top 3 similar chunks
   via cosine similarity
3. Builds the final answer

This generation step branches on MOCK_LLM.

Mock mode (graded baseline): takes the first 200 characters of the top
matched chunk and returns "Based on the retrieved context: [snippet]".
Sources are the ChromaDB document IDs of the retrieved chunks. Confidence
is 1.0. No LLM call.

Real LLM mode (optional extension): the top 3 chunks are joined and
passed to ZEPTO_SUPPORT_PROMPT in llm_utils.py, then sent to Groq.
The raw JSON response is validated against FinalAnswerSchema. If it
fails, the code retries up to 2 more times with a corrective instruction
before returning an error response.

For general questions, direct_answer returns the fixed canned string
"I can only answer questions about Zepto policies right now." in mock
mode with empty sources and confidence 1.0. In real LLM mode it calls
ask_llm_direct from llm_utils.py with no retrieval context.

### Final Output - Pydantic Schema (schemas.py)

Every node produces data that ends up in FinalAnswerSchema with three
fields: answer (string), sources (list of ChromaDB document ID strings),
and confidence (float 0 to 1). This is nested inside QueryResponse
which also includes the query text, the classified intent, and the list
of retrieved document texts.

---

## Project Files

docs/doc_01.txt - Delivery Policy
docs/doc_02.txt - Returns and Refunds
docs/doc_03.txt - Membership Tiers
docs/doc_04.txt - Order Tracking
docs/doc_05.txt - Order Cancellation Policy
docs/doc_06.txt - Damaged or Missing Items
docs/doc_07.txt - Gift Cards
docs/doc_08.txt - Customer Support Hours

schemas.py     - pydantic models and GraphState TypedDict
embed_store.py - sentence encoder setup and chromadb ingestion
llm_utils.py   - groq helpers with retry logic and the prompt template
nodes.py       - classify_intent, retrieve_and_answer, direct_answer
graph.py       - builds and compiles the LangGraph StateGraph
app.py         - FastAPI app with the POST /ask endpoint
requirements.txt - all required python packages
Dockerfile     - container build config, exposes port 7860
README.md      - this file

---

## MOCK_LLM Toggle Summary

Stage 1 ingestion            - always the same, no branching
Stage 2 embedding + storage  - always the same, no branching
Stage 3 intent classification - mock: keyword check vs real: Groq API call
Stage 4 answer generation    - mock: 200 char snippet vs real: Groq API call

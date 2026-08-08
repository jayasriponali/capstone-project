from fastapi import FastAPI
from models import QueryRequest, QueryResponse, FinalAnswerSchema
from ingest import support_graph, GraphState

# create the FastAPI app with a title
zepto_app = FastAPI(title="Zepto Customer Support Assistant API")


@zepto_app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest) -> QueryResponse:
    # set up the starting state for the langgraph pipeline
    start_state: GraphState = {
        "query": request.query,
        "intent": "general_question",
        "retrieved_docs": [],
        "answer": {}
    }

    # run the query through the langgraph state machine
    result = support_graph.invoke(start_state)

    # convert the answer dict into the pydantic schema
    raw_answer = result.get("answer", {})
    if isinstance(raw_answer, dict):
        final_resp = FinalAnswerSchema(**raw_answer)
    else:
        final_resp = FinalAnswerSchema(answer=str(raw_answer), sources=[], confidence=1.0)

    return QueryResponse(
        query=result["query"],
        intent=result["intent"],
        retrieved_docs=result.get("retrieved_docs", []),
        answer=final_resp
    )

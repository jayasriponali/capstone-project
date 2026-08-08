from fastapi import FastAPI
from schemas import QueryRequest, QueryResponse, FinalAnswerSchema, GraphState
from graph import zepto_graph

zepto_app = FastAPI(title="Zepto Customer Support Assistant API")


@zepto_app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest) -> QueryResponse:
    # set up the starting state for the langgraph pipeline
    initial_state: GraphState = {
        "query": request.query,
        "intent": "general_question",
        "retrieved_docs": [],
        "answer": {}
    }

    # run the query through the full graph
    result = zepto_graph.invoke(initial_state)

    # wrap the answer dict in the pydantic schema
    answer_dict = result.get("answer", {})
    if isinstance(answer_dict, dict):
        parsed_answer = FinalAnswerSchema(**answer_dict)
    else:
        parsed_answer = FinalAnswerSchema(answer=str(answer_dict), sources=[], confidence=1.0)

    return QueryResponse(
        query=result["query"],
        intent=result["intent"],
        retrieved_docs=result.get("retrieved_docs", []),
        answer=parsed_answer
    )

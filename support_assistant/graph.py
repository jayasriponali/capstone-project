from langgraph.graph import StateGraph, START, END
from schemas import GraphState
from nodes import classify_intent, retrieve_and_answer, direct_answer, pick_next_step

# build the state machine and register all three nodes
graph_builder = StateGraph(GraphState)

graph_builder.add_node("classify_intent",    classify_intent)
graph_builder.add_node("retrieve_and_answer", retrieve_and_answer)
graph_builder.add_node("direct_answer",       direct_answer)

# every query starts at classify_intent
graph_builder.add_edge(START, "classify_intent")

# after classification the conditional edge picks the right path
graph_builder.add_conditional_edges(
    "classify_intent",
    pick_next_step,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer":       "direct_answer"
    }
)

# both answer nodes lead to the end
graph_builder.add_edge("retrieve_and_answer", END)
graph_builder.add_edge("direct_answer",        END)

# compile once so the graph is ready to handle requests
zepto_graph = graph_builder.compile()


if __name__ == "__main__":
    test_input: GraphState = {
        "query": "How do I cancel my order?",
        "intent": "general_question",
        "retrieved_docs": [],
        "answer": ""
    }
    output = zepto_graph.invoke(test_input)
    print("\nResult:", output.get("answer"))

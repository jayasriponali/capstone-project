from pydantic import BaseModel, Field
from typing import List, Literal, Union
from typing_extensions import TypedDict


# -----------------------------------------------------------------------
# Pydantic models for the FastAPI request and response
# -----------------------------------------------------------------------

# what the customer sends in via POST /ask
class QueryRequest(BaseModel):
    query: str


# structured answer that every node must produce at the end
# sources holds the chromadb document IDs for retrieved chunks
class FinalAnswerSchema(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# full response returned by the /ask endpoint
class QueryResponse(BaseModel):
    query: str
    intent: str
    retrieved_docs: List[str]
    answer: FinalAnswerSchema


# -----------------------------------------------------------------------
# LangGraph state - holds all the data as it moves through the graph
# -----------------------------------------------------------------------
class GraphState(TypedDict):
    query: str
    intent: Literal["policy_question", "general_question"]
    retrieved_docs: list[str]
    answer: Union[dict, str]

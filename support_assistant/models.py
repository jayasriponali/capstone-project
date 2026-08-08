from pydantic import BaseModel, Field
from typing import List


# this holds the final answer the assistant sends back to the user
class FinalAnswerSchema(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# this is the request body that the user sends to the /ask endpoint
class QueryRequest(BaseModel):
    query: str


# this is the full response we send back to the user
class QueryResponse(BaseModel):
    query: str
    intent: str
    retrieved_docs: List[str]
    answer: FinalAnswerSchema

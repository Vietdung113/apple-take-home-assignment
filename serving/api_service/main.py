"""FastAPI app — POST /summarize."""

from fastapi import FastAPI
from pydantic import BaseModel

from api_service.agents.graph import pipeline

app = FastAPI(title="Summarization Service")


class SummarizeRequest(BaseModel):
    document: str


class SummarizeResponse(BaseModel):
    summary: str


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(req: SummarizeRequest):
    result = await pipeline.ainvoke({"document": req.document})
    return SummarizeResponse(summary=result["final_summary"])


@app.get("/health")
def health():
    return {"status": "ok"}

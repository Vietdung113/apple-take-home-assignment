"""FastAPI app — POST /summarize."""

import os

from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

# Connect to Phoenix via gRPC (port 4317)
PHOENIX_GRPC = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:4317")
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=PHOENIX_GRPC)))
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

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

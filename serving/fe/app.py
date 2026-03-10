import os

import gradio as gr
import httpx

API_URL = os.environ.get("API_URL", "http://localhost:8001/summarize")


def summarize(document: str) -> str:
    if not document.strip():
        return "Please paste a document to summarize."
    try:
        resp = httpx.post(API_URL, json={"document": document}, timeout=300)
        resp.raise_for_status()
        return resp.json()["summary"]
    except httpx.ConnectError:
        return f"Error: Cannot connect to API at {API_URL}. Is the API service running?"
    except httpx.HTTPStatusError as e:
        return f"Error: API returned {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return f"Error: {e}"


with gr.Blocks(title="Summarization Demo") as demo:
    gr.Markdown("# Summarization Demo")
    gr.Markdown("Paste a document and click **Summarize** to generate a summary using the agentic pipeline.")
    document = gr.Textbox(lines=20, label="Document", placeholder="Paste your document here...")
    btn = gr.Button("Summarize", variant="primary")
    summary = gr.Textbox(lines=10, label="Summary")
    btn.click(fn=summarize, inputs=[document], outputs=[summary], api_name="summarize")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")

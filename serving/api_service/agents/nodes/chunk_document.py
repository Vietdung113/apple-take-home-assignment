"""Node: split document into chunks by sentence boundaries."""

import re

from api_service.agents.state import SummaryState

CHUNK_SIZE = 6000       # chars per chunk
OVERLAP_SENTENCES = 2   # overlap sentences between chunks
LONG_DOC_THRESHOLD = 60000  # ~15K tokens (1 token ≈ 4 chars)

# Sentence splitter: split on ". " followed by uppercase, or common patterns
_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    return _SENT_RE.split(text)


def _chunk_long_document(doc: str) -> list[str]:
    """Split long document into chunks."""
    sentences = _split_sentences(doc)

    # If sentence splitting fails (e.g. no periods), fall back to char-based
    if len(sentences) <= 1:
        # Split by spaces at ~CHUNK_SIZE boundaries
        words = doc.split()
        chunks = []
        chunk_words = []
        chunk_len = 0
        for w in words:
            if chunk_len + len(w) + 1 > CHUNK_SIZE and chunk_words:
                chunks.append(" ".join(chunk_words))
                chunk_words = chunk_words[-20:]  # overlap ~20 words
                chunk_len = sum(len(w) + 1 for w in chunk_words)
            chunk_words.append(w)
            chunk_len += len(w) + 1
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        return chunks

    # Sentence-based chunking
    chunks = []
    current_sents: list[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent) + 1

        if current_len + sent_len > CHUNK_SIZE and current_sents:
            chunks.append(" ".join(current_sents))
            # Overlap: keep last N sentences
            current_sents = current_sents[-OVERLAP_SENTENCES:]
            current_len = sum(len(s) + 1 for s in current_sents)

        current_sents.append(sent)
        current_len += sent_len

    if current_sents:
        chunks.append(" ".join(current_sents))

    return chunks


def chunk_document_node(state: SummaryState) -> dict:
    """Determine routing: direct summarize or chunked pipeline."""
    doc = state["document"]
    doc_len = len(doc)
    doc_words = len(doc.split())
    target_words = max(80, doc_words // 18)

    # Check if document is long enough to require chunking
    is_long = doc_len >= LONG_DOC_THRESHOLD

    if not is_long:
        # Short/medium doc: use direct summarize path
        print(f"  Route: DIRECT (doc {doc_len:,} chars < {LONG_DOC_THRESHOLD:,} threshold)")
        return {
            "chunks": [],
            "target_words": target_words,
            "is_long_document": False,
        }
    else:
        # Long doc: use extract pipeline
        chunks = _chunk_long_document(doc)
        avg_chunk_len = sum(len(c) for c in chunks) // len(chunks) if chunks else 0
        print(f"  Route: EXTRACT ({len(chunks)} chunks, "
              f"doc {doc_len:,} chars, avg {avg_chunk_len:,} chars/chunk)")
        return {
            "chunks": chunks,
            "target_words": target_words,
            "is_long_document": True,
        }

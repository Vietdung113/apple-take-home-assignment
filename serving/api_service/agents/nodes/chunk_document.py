"""Node: split document into chunks by sentence boundaries."""

import re

from api_service.agents.state import SummaryState

CHUNK_SIZE = 6000       # chars per chunk
OVERLAP_SENTENCES = 2   # overlap sentences between chunks

# Sentence splitter: split on ". " followed by uppercase, or common patterns
_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    return _SENT_RE.split(text)


def chunk_document_node(state: SummaryState) -> dict:
    """Split document into chunks of ~6000 chars at sentence boundaries."""
    doc = state["document"]
    doc_words = len(doc.split())
    target_words = max(80, doc_words // 18)

    # Short docs: no chunking needed
    if len(doc) <= CHUNK_SIZE * 1.3:
        print(f"  Chunk: 1 chunk (doc {len(doc):,} chars, no split needed)")
        return {"chunks": [doc], "target_words": target_words}

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

        print(f"  Chunk: {len(chunks)} chunks from {len(doc):,} chars "
              f"(word-based split)")
        return {"chunks": chunks, "target_words": target_words}

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

    print(f"  Chunk: {len(chunks)} chunks from {len(doc):,} chars "
          f"(avg {sum(len(c) for c in chunks) // len(chunks):,} chars/chunk)")

    return {"chunks": chunks, "target_words": target_words}

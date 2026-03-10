"""Agent prompts — loaded from centralized config (finetuning/config/prompts.yaml).

All prompts flow through prompt_loader so there's a single source of truth.
"""

import sys
from pathlib import Path

# Add finetuning config to path so prompt_loader is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "finetuning" / "config"))
from prompt_loader import (  # noqa: E402, F401
    get_direct_summarize_prompt,
    get_summarize_chunk_prompt,
    get_merge_summaries_prompt,
    get_system_prompt,
    get_generation_params,
)

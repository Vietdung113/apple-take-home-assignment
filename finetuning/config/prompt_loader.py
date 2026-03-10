"""Load prompts from centralized config (prompts.yaml).

Single source of truth for training, eval, and serving.
"""

from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).parent / "prompts.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)


# ── Core prompts ─────────────────────────────────────────────────────────

def get_system_prompt() -> str:
    return CONFIG["system_prompt"].strip()


def get_user_instruction() -> str:
    return CONFIG["user_instruction"]


def get_summary_instruction() -> str:
    return CONFIG["summary_instruction"]


def get_generation_params() -> dict:
    return CONFIG["generation"]


# ── Training prompts ─────────────────────────────────────────────────────

def get_training_prompt_base_model(document: str, summary: str) -> str:
    """Base model training format: plain text continuation.

    IMPORTANT: Caller must append tokenizer.eos_token to teach model when to stop.
    Without EOS token, model will ramble indefinitely during inference.
    """
    return (
        f"{CONFIG['system_prompt']}\n"
        f"{CONFIG['user_instruction']}"
        f"{document}\n\n"
        f"{CONFIG['summary_instruction']}\n\n"
        f"Expert Summary:\n{summary}"
    )


def get_training_prompt_instruct_model(document: str, summary: str) -> dict:
    """Instruct model training format: chat messages."""
    user_content = f"{CONFIG['user_instruction']}{document}\n\n{CONFIG['summary_instruction']}"
    return {
        "messages": [
            {"role": "system", "content": CONFIG["system_prompt"].strip()},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": summary},
        ]
    }


# ── Eval / Inference prompts ─────────────────────────────────────────────

def get_inference_prompt_base_model(document: str) -> str:
    """Base model inference format: plain text."""
    return (
        f"{CONFIG['system_prompt']}\n"
        f"{CONFIG['user_instruction']}"
        f"{document}\n\n"
        f"{CONFIG['summary_instruction']}\n\n"
        f"Expert Summary:\n"
    )


def get_inference_prompt_instruct_model(document: str) -> list[dict]:
    """Instruct model inference format: chat messages."""
    user_content = f"{CONFIG['user_instruction']}{document}\n\n{CONFIG['summary_instruction']}"
    return [
        {"role": "system", "content": CONFIG["system_prompt"].strip()},
        {"role": "user", "content": user_content},
    ]


def get_judge_prompt(document: str, reference: str, generated: str) -> str:
    """LLM-as-judge evaluation prompt."""
    return CONFIG["judge_prompt"].format(
        document=document,
        reference=reference,
        generated=generated,
    )


# ── Agent prompts (serving pipeline) ─────────────────────────────────────

def get_direct_summarize_prompt(document: str) -> str:
    """Direct summarize: user_instruction + document + summary_instruction."""
    return f"{CONFIG['user_instruction']}{document}\n\n{CONFIG['summary_instruction']}"


def get_summarize_chunk_prompt(chunk: str) -> str:
    return f"{CONFIG['agent']['summarize_chunk']}\n\nSection:\n{chunk}"


def get_merge_summaries_prompt(section_summaries: str) -> str:
    return f"{CONFIG['agent']['merge_summaries']}\n\nSection Summaries:\n{section_summaries}"


def get_extract_facts_prompt(chunk: str) -> str:
    return f"{CONFIG['agent']['extract_facts']}\n\nSection:\n{chunk}"


def get_synthesize_from_facts_prompt(facts: str) -> str:
    return f"{CONFIG['agent']['synthesize_from_facts']}\n\nExtracted Facts:\n{facts}"


def get_extract_outline_prompt(chunk: str) -> str:
    return f"{CONFIG['agent']['extract_outline']}\n\n{chunk}"


def get_extract_facts_with_context_prompt(
    structure: str, section_num: int, total_sections: int,
    current_topic: str, chunk: str,
) -> str:
    return (
        f"{CONFIG['agent']['extract_facts_with_context']}\n\n"
        f"DOCUMENT STRUCTURE:\n{structure}\n\n"
        f"CURRENT SECTION ({section_num}/{total_sections}): {current_topic}\n\n"
        f"SECTION CONTENT:\n{chunk}"
    )


def get_initial_summarize_prompt(chunk: str) -> str:
    return f"{CONFIG['agent']['initial_summarize']}\n\n{chunk}"


def get_refine_summary_prompt(current_summary: str, next_chunk: str) -> str:
    return (
        f"{CONFIG['agent']['refine_summary']}\n\n"
        f"SUMMARY SO FAR:\n{current_summary}\n\n"
        f"NEW SECTION TO INTEGRATE:\n{next_chunk}"
    )

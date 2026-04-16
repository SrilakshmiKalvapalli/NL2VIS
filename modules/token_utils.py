# modules/token_utils.py
import tiktoken

def count_tokens(text: str) -> int:
    """
    Token counting compatible with Visistant paper.

    Visistant used GPT-3.5-Turbo, which internally uses the
    cl100k_base tokenizer. We directly use cl100k_base to
    ensure identical token counts without hard-coding
    deprecated model names.
    """
    if not text:
        return 0

    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

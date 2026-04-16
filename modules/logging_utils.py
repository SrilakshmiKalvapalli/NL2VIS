# modules/logging_utils.py
import os
import csv
import time
from datetime import datetime
from modules.token_utils import count_tokens


LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "interaction_logs.csv")


def ensure_log_file():
    """
    Ensure logs directory and CSV file exist, create header if needed.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "dataset_name",
                "dataset_hash",
                "query",
                "context_length_chars",
                "prompt_length_chars",
                "response_length_chars",
                "prompt_tokens",
                "response_tokens",
                "chart_type",
                "x_col",
                "y_col",
                "validation_ok",
                "error_type",
                "latency_seconds",
            ])


def log_interaction(
    dataset_name: str,
    dataset_hash: str,
    query: str,
    context: str,
    prompt: str,
    response_text: str,
    chart_type: str | None,
    x_col: str | None,
    y_col: str | None,
    validation_ok: bool,
    error_type: str | None,
    start_time: float,
    end_time: float,
):
    """
    Append one interaction row to logs/interaction_logs.csv with
    character lengths, token counts, and timing.
    """
    ensure_log_file()

    context_len = len(context or "")
    prompt_len = len(prompt or "")
    resp_len = len(response_text or "")

    prompt_tokens = count_tokens(prompt or "")
    response_tokens = count_tokens(response_text or "")

    latency = max(0.0, end_time - start_time)

    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(),
            dataset_name,
            dataset_hash,
            query,
            context_len,
            prompt_len,
            resp_len,
            prompt_tokens,
            response_tokens,
            chart_type or "",
            x_col or "",
            y_col or "",
            "yes" if validation_ok else "no",
            error_type or "",
            f"{latency:.4f}",
        ])

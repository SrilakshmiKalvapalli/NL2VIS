# modules/chart_planner.py
from modules.insights import client  # reuse same genai.Client
import textwrap
import json
import re


def plan_chart(df_head_csv: str, query: str, return_meta: bool = False):
    """
    Ask Gemini to choose the MOST SUITABLE chart type and columns
    for analysis (line, bar, scatter, hist, box, pie, area, bubble).

    If return_meta=True, returns:
        (chart_type, x_col, y_col, full_prompt, raw_response_text)
    Otherwise returns:
        (chart_type, x_col, y_col)
    """

    prompt = textwrap.dedent("""
    You are an expert data visualization analyst.

    Your job:
    - Look at the SAMPLE of the dataset (in CSV).
    - Read the user's question.
    - Decide which ONE chart type is most suitable to ANALYZE and UNDERSTAND
      the pattern the user is asking about.

    Choose the chart that best matches the analytical goal:
    - line: trends or change over time.
    - bar: compare categories.
    - scatter: relationship between 2 numeric variables.
    - hist: distribution of a numeric variable.
    - box: distribution + outliers.
    - pie: parts of a whole with few categories.
    - area: cumulative change over time.
    - bubble: 3 numeric variables (x, y, size) for relationship.

    You MUST answer ONLY in this exact JSON format (no extra text):

    {
      "chart_type": "<one of: line, bar, scatter, hist, box, pie, area, bubble>",
      "x_col": "<column name or null>",
      "y_col": "<column name or null>"
    }

    Rules:
    - Use the dataset columns exactly as they appear in the header.
    - For line/area: x_col is usually a time-like column (date/year), y_col numeric.
    - For bar: x_col categorical, y_col numeric.
    - For scatter/bubble: x_col and y_col numeric.
    - For hist/box: x_col numeric, y_col must be null.
    - For pie: x_col categorical, y_col numeric.
    - Only pick bar if it is really the best for the analysis goal.
    """)

    user_prompt = f"""
    Here is the first few rows of the dataset in CSV format:

    {df_head_csv}

    User question:
    {query}
    """

    full_prompt = f"{prompt}\n\n{user_prompt}"

    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=full_prompt,
    )

    raw_text = (response.text or "").strip()
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError("Chart planner: could not parse JSON from model response")

    config = json.loads(match.group(0))
    chart_type = (config.get("chart_type") or "").strip().lower() or None
    x_col = config.get("x_col")
    y_col = config.get("y_col")

    # normalize "null"
    if isinstance(x_col, str) and x_col.lower() == "null":
        x_col = None
    if isinstance(y_col, str) and y_col.lower() == "null":
        y_col = None

    if return_meta:
        return chart_type, x_col, y_col, full_prompt, raw_text
    return chart_type, x_col, y_col


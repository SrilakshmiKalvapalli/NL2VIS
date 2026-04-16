# app.py

import time
import hashlib

import pandas as pd
import streamlit as st

from config.gemini_config import configure_gemini
from modules.embeddings import embed_texts, build_faiss_index
from modules.retriever import search_index
from modules.insights import generate_insights
from modules.viz_generator import generate_plot
from modules.chart_planner import plan_chart
from modules.logging_utils import log_interaction
from modules.token_utils import count_tokens          # ← ADDED for token metrics

print("🔥 USING CORRECT EMBEDDINGS FILE 🔥")


# ---------------- Streamlit Setup ----------------
st.set_page_config(layout="wide")
st.title("NL → Visualization + Insights with Gemini LLM")


# Shared Gemini client (embeddings + retrieval)
client = configure_gemini()  # should return genai.Client()


# ---------------- Session Cache ----------------
if "chart_cache" not in st.session_state:
    st.session_state.chart_cache = {}  # (dataset_hash, query) -> (chart_type, x_col, y_col)

if "current_ds_hash" not in st.session_state:
    st.session_state.current_ds_hash = None


def get_dataset_hash(df: pd.DataFrame) -> str:
    """Hash first 50 rows + columns to identify dataset version."""
    csv_sample = df.head(50).to_csv(index=False)
    return hashlib.sha256(csv_sample.encode("utf-8")).hexdigest()


# ---------------- UI ----------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
query = st.text_input(
    "Ask a question about the dataset",
    placeholder="e.g., Which movies were financially successful?",
)


if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully")
    st.dataframe(df.head())

    dataset_name = uploaded_file.name
    ds_hash = get_dataset_hash(df)

    # If dataset changed, reset cache
    if st.session_state.current_ds_hash != ds_hash:
        st.session_state.chart_cache = {}
        st.session_state.current_ds_hash = ds_hash

    # -------- Build RAG Corpus --------
    texts = []
    for col in df.columns:
        # Convert to string and take a small sample
        col_data = df[col].astype(str).head(10).tolist()
        col_text = f"Column '{col}' (dtype={df[col].dtype}): " + ", ".join(col_data)
        texts.append(col_text)

    if not texts:
        st.error("No columns found in the dataset.")
    else:
        # -------- Build FAISS Index --------
        try:
            embeddings = embed_texts(texts, client)
            index = build_faiss_index(embeddings)
        except Exception as e:
            st.error(f"Error creating embeddings/index: {e}")
            st.stop()

        if query:
            start_time = time.time()

            # -------- RAG Retrieval --------
            context_snippets = search_index(
                query=query,
                client=client,
                index=index,
                texts=texts,
                top_k=3,
            )
            context = "\n".join(context_snippets)

            # -------- Insights --------
            st.subheader("Insights")
            insights = generate_insights(context, query)
            st.write(insights)

            # -------- Visualization --------
            st.subheader("Visualization")

            cache_key = (ds_hash, query.strip())

            full_prompt = ""
            planner_response_text = ""
            chart_type = x_col = y_col = None
            validation_ok = False
            error_type = None

            # -------- Chart Planning (Cached) --------
            if cache_key in st.session_state.chart_cache:
                chart_type, x_col, y_col = st.session_state.chart_cache[cache_key]
                full_prompt = "[CACHED] Chart configuration reused"
            else:
                df_head_csv = df.head(20).to_csv(index=False)
                try:
                    chart_type, x_col, y_col, full_prompt, planner_response_text = plan_chart(
                        df_head_csv, query, return_meta=True
                    )
                    st.session_state.chart_cache[cache_key] = (chart_type, x_col, y_col)
                except Exception as e:
                    error_type = f"planner_error: {e}"
                    st.error(f"Chart planning failed: {e}")

            # -------- Validation --------
            if chart_type in ["hist", "box"]:
                validation_ok = x_col is not None
            elif chart_type == "pie":
                validation_ok = x_col is not None and y_col is not None
            else:
                validation_ok = x_col is not None and y_col is not None

            if validation_ok:
                missing_cols = []
                if x_col and x_col not in df.columns:
                    missing_cols.append(x_col)
                if y_col and y_col not in df.columns:
                    missing_cols.append(y_col)

                if missing_cols:
                    validation_ok = False
                    error_type = f"missing_columns: {', '.join(missing_cols)}"

            # -------- Plot --------
            if validation_ok:
                try:
                    fig = generate_plot(df, x_col, y_col, chart_type)
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown(
                        f"Gemini selected a **{chart_type}** chart "
                        f"with X = `{x_col}` and Y = `{y_col}`."
                    )
                except Exception as e:
                    validation_ok = False
                    error_type = f"plot_error: {e}"
                    st.error(f"Plot generation failed: {e}")
            else:
                if error_type is None:
                    error_type = "invalid_chart_config"
                st.info("No valid chart configuration could be inferred.")

            end_time = time.time()
            latency_ms = round((end_time - start_time) * 1000, 2)

            # -------- Token Metrics --------          ← ADDED BLOCK
            context_tokens  = count_tokens(context or "")
            prompt_tokens   = count_tokens(full_prompt or "")
            response_tokens = count_tokens(insights or "")
            total_tokens    = prompt_tokens + response_tokens

            # -------- Performance Indicator --------   ← UPDATED caption
            st.caption(
                f"⏱ Response Time: {latency_ms} ms | "
                f"✔ Valid Chart: {validation_ok} | "
                f"🔢 Prompt Tokens: {prompt_tokens} | "
                f"📦 Total Tokens: {total_tokens}"
            )

            # -------- Research Metrics (UI) --------   ← UPDATED expander
            with st.expander("📊 Evaluation Metrics (Research)"):
                st.markdown("**Chart Configuration**")
                st.write({
                    "Chart Type"    : chart_type,
                    "X Column"      : x_col,
                    "Y Column"      : y_col,
                    "Validation OK" : validation_ok,
                    "Latency (ms)"  : latency_ms,
                })
                st.markdown("**Token Metrics (RAG Efficiency)**")
                st.write({
                    "Context Tokens (RAG retrieved)" : context_tokens,
                    "Prompt Tokens (sent to Gemini)" : prompt_tokens,
                    "Response Tokens (from Gemini)"  : response_tokens,
                    "Total Tokens"                   : total_tokens,
                })

            # -------- SAFE LOGGING (CRITICAL) --------
            try:
                log_interaction(
                    dataset_name=dataset_name,
                    dataset_hash=ds_hash,
                    query=query,
                    context=context,
                    prompt=full_prompt,
                    response_text=insights,   # log real LLM output
                    chart_type=chart_type,
                    x_col=x_col,
                    y_col=y_col,
                    validation_ok=validation_ok,
                    error_type=error_type,
                    start_time=start_time,
                    end_time=end_time,
                )
            except Exception as e:
                st.warning("⚠ Logging skipped due to tokenization/logging issue.")
                print("Logging error:", e)

else:
    st.info("Upload a CSV file to start.")








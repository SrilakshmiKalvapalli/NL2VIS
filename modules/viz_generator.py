# modules/viz_generator.py
import plotly.express as px
import pandas as pd

def generate_plot(df: pd.DataFrame, x_col: str, y_col: str | None, chart_type: str):
    t = (chart_type or "scatter").lower()

    if t == "bar":
        fig = px.bar(df, x=x_col, y=y_col, title=f"Bar chart: {x_col} vs {y_col}")
    elif t == "line":
        fig = px.line(df, x=x_col, y=y_col, title=f"Line chart: {x_col} vs {y_col}")
    elif t == "hist":
        fig = px.histogram(df, x=x_col, title=f"Histogram of {x_col}")
    elif t == "box":
        fig = px.box(df, x=x_col, title=f"Box plot of {x_col}")
    elif t == "pie":
        fig = px.pie(df, names=x_col, values=y_col, title=f"Pie chart of {x_col}")
    elif t == "area":
        fig = px.area(df, x=x_col, y=y_col, title=f"Area chart: {x_col} vs {y_col}")
    elif t == "bubble":
        fig = px.scatter(
            df, x=x_col, y=y_col, size=y_col,
            title=f"Bubble chart: {x_col} vs {y_col}"
        )
    else:  # default scatter
        fig = px.scatter(df, x=x_col, y=y_col, title=f"Scatter: {x_col} vs {y_col}")

    return fig


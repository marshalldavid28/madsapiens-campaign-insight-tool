from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import openai
from io import BytesIO
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🔑 OpenAI API Key loaded?", bool(openai.api_key))

# Helper for calculating metrics
def safe_div(a, b):
    return round(a / b, 2) if b else 0

# Group summary builder
def build_group_summary(df, group_by_col, label):
    if group_by_col not in df.columns:
        return ""

    df[group_by_col] = df[group_by_col].fillna("Unknown")
    grouped = df.groupby(group_by_col).agg({
        'Impressions': 'sum',
        'Clicks': 'sum',
        'Spend': 'sum',
        'Total Conversions': 'sum'
    }).reset_index()

    grouped['CTR (%)'] = (grouped['Clicks'] / grouped['Impressions']) * 100
    grouped['CPM (SGD)'] = (grouped['Spend'] / grouped['Impressions']) * 1000
    grouped['CPC (SGD)'] = grouped['Spend'] / grouped['Clicks']
    grouped['Conversion Rate (%)'] = (grouped['Total Conversions'] / grouped['Impressions']) * 100

    grouped = grouped.fillna(0)
    grouped = grouped[grouped['Impressions'] >= 1000]  # Ignore tiny rows
    grouped = grouped.sort_values(by='Spend', ascending=False).head(5)

    parts = []
    for _, row in grouped.iterrows():
        summary = f"{label}: {row[group_by_col]}\n"
        summary += f"Impressions: {int(row['Impressions'])}, Clicks: {int(row['Clicks'])}, CTR: {row['CTR (%)']:.2f}%\n"
        summary += f"Spend: SGD {row['Spend']:.2f}, CPM: SGD {row['CPM (SGD)']:.2f}, CPC: SGD {row['CPC (SGD)']:.2f}\n"
        summary += f"Conversions: {int(row['Total Conversions'])}, Conversion Rate: {row['Conversion Rate (%)']:.2f}%"
        parts.append(summary)

    return "\n\n".join(parts)

# ---------- Main endpoint ----------
@app.post("/generate-insights/")
async def generate_insights(
    file: UploadFile = File(...),
    objective: str = Form(...),
    ctr_target: float = Form(...),
    cpm_target: float = Form(...),
    budget: float = Form(...),
    flight: str = Form(...),
    primary_metric: str = Form(...),
    secondary_metric: str = Form(None),
    breakdown_1: str = Form(None),
    breakdown_2: str = Form(None),
    breakdown_3: str = Form(None)
):
    try:
        start = time.time()
        contents = await file.read()
        df = pd.read_excel(BytesIO(contents))
        print("✅ File loaded:", df.shape)

        # Column normalization
        df.columns = [col.strip().title() for col in df.columns]

        # Rename common column variants
        column_aliases = {
            "Spends": "Spend",
            "Total Conversion": "Total Conversions"
        }
        df = df.rename(columns={k: v for k, v in column_aliases.items() if k in df.columns})

        # Check essential columns
        essential = ["Impressions", "Clicks", "Spend"]
        for col in essential:
            if col not in df.columns:
                return JSONResponse(status_code=400, content={"error": f"Missing column: {col}"})

        if "Total Conversions" not in df.columns:
            df["Total Conversions"] = 0

        # Replace missing values
        df["Spend"] = df["Spend"].replace(0, 0.01)
        df["Clicks"] = df["Clicks"].replace(0, 1)
        df.fillna(0, inplace=True)

        # Totals
        total_impressions = df["Impressions"].sum()
        total_clicks = df["Clicks"].sum()
        total_spend = df["Spend"].sum()
        total_conversions = df["Total Conversions"].sum()

        ctr = safe_div(total_clicks * 100, total_impressions)
        cpm = safe_div(total_spend * 1000, total_impressions)
        cpc = safe_div(total_spend, total_clicks)
        conv_rate = safe_div(total_conversions * 100, total_impressions)
        cost_per_conv = safe_div(total_spend, total_conversions)

        # Summaries
        breakdowns = [b for b in [breakdown_1, breakdown_2, breakdown_3] if b]
        summaries = []
        for b in breakdowns:
            summaries.append(build_group_summary(df, b, b))

        breakdown_text = "\n\n".join([f"### {label}\n{text}" for label, text in zip(breakdowns, summaries) if text])
        breakdown_headers = "".join([f"## {b} Analysis\n" for b in breakdowns])

        # Final prompt
        prompt = (
            f"You are a professional paid media strategist reporting on a DV360 Display campaign.\n\n"
            f"Your tone is confident, data-driven, and insightful. Use the following structure:\n\n"
            f"## Executive Summary\n"
            f"## Performance vs KPIs\n"
            f"## Conversion Analysis\n"
            f"{breakdown_headers}"
            f"## Strategic Observations & Recommendations\n\n"
            f"## CAMPAIGN BRIEF\n"
            f"- Objective: {objective}\n"
            f"- CTR Target: {ctr_target}%\n"
            f"- CPM Target: SGD {cpm_target}\n"
            f"- Budget: SGD {budget}\n"
            f"- Flight: {flight}\n"
            f"- Primary Metric: {primary_metric}\n"
            f"- Secondary Metric: {secondary_metric or 'None'}\n\n"
            f"## OVERALL PERFORMANCE\n"
            f"- Impressions: {total_impressions:,}\n"
            f"- Clicks: {total_clicks:,}\n"
            f"- CTR: {ctr:.2f}%\n"
            f"- Spend: SGD {total_spend:,.2f}\n"
            f"- CPM: SGD {cpm:,.2f}\n"
            f"- CPC: SGD {cpc:,.2f}\n"
            f"- Conversions: {total_conversions:,}\n"
            f"- Conversion Rate: {conv_rate:.2f}%\n"
            f"- Cost per Conversion: SGD {cost_per_conv:,.2f}\n\n"
            f"{breakdown_text}"
        )

        print("📤 Prompt ready, sending to GPT...")
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            temperature=0.85,
            messages=[
                {"role": "system", "content": "You write strategic, confident campaign insights."},
                {"role": "user", "content": prompt}
            ]
        )

        report_text = response.choices[0].message.content
        print("✅ GPT response received")
        print(f"⏱️ Total time: {round(time.time() - start, 2)}s")
        return JSONResponse(content={"report": report_text})

    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})


# ---------- Chat endpoint ----------
class InteractionRequest(BaseModel):
    insight: str
    user_prompt: str
    mode: str  # "ask" or "edit"

@app.post("/interact-insight/")
async def interact_with_insight(request: InteractionRequest):
    try:
        print("💬 Chat interaction received:", request.dict())
        instruction = (
            "You are a paid media analyst. "
            "If mode is 'ask', answer the user’s question using the insight text only. "
            "If mode is 'edit', revise or improve the insight based on the user's prompt."
        )

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": f"--- Original Insight ---\n{request.insight}"},
            {"role": "user", "content": f"--- User Prompt ---\n{request.user_prompt}"},
            {"role": "user", "content": f"Mode: {request.mode}"}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.75,
            messages=messages
        )

        result = response.choices[0].message.content
        return JSONResponse(content={"result": result})

    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import openai
from io import BytesIO
import os
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
print("🔑 OpenAI API key loaded:", bool(openai.api_key))

# ---------- Column Normalization ----------
COLUMN_ALIASES = {
    "spend": "Spend",
    "spends": "Spend",
    "click": "Clicks",
    "clicks": "Clicks",
    "impression": "Impressions",
    "impressions": "Impressions",
    "total conversions": "Total Conversions",
    "conversions": "Total Conversions",
}


def normalize_columns(df):
    df.columns = [COLUMN_ALIASES.get(col.lower().strip(), col.strip()) for col in df.columns]
    return df


def build_group_summary(df, group_by_col):
    if group_by_col not in df.columns:
        return ""

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
    grouped = grouped.fillna(0).sort_values(by='Spend', ascending=False).head(5)

    summaries = []
    for _, row in grouped.iterrows():
        summary = f"{group_by_col}: {row[group_by_col]}\n"
        summary += f"Impressions: {int(row['Impressions'])}, Clicks: {int(row['Clicks'])}, CTR: {row['CTR (%)']:.2f}%\n"
        summary += f"Spend: SGD {row['Spend']:.2f}, CPM: SGD {row['CPM (SGD)']:.2f}, CPC: SGD {row['CPC (SGD)']:.2f}\n"
        summary += f"Conversions: {int(row['Total Conversions'])}, Conversion Rate: {row['Conversion Rate (%)']:.2f}%"
        summaries.append(summary)

    return "\n\n".join(summaries)


# ---------- MAIN INSIGHTS ENDPOINT ----------
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
        contents = await file.read()
        df = pd.read_excel(BytesIO(contents))
        df = normalize_columns(df)
        print("✅ Data loaded with columns:", df.columns.tolist())

        # Required columns check
        required = ["Impressions", "Clicks", "Spend"]
        for col in required:
            if col not in df.columns:
                return JSONResponse(status_code=400, content={"error": f"Missing column: {col}"})

        if "Total Conversions" not in df.columns:
            df["Total Conversions"] = 0

        # Totals
        total_impressions = df["Impressions"].sum()
        total_clicks = df["Clicks"].sum()
        total_spend = df["Spend"].sum()
        total_conversions = df["Total Conversions"].sum()

        ctr = round((total_clicks / total_impressions) * 100, 2) if total_impressions else 0
        cpm = round((total_spend / total_impressions) * 1000, 2) if total_impressions else 0
        cpc = round(total_spend / total_clicks, 2) if total_clicks else 0
        conv_rate = round((total_conversions / total_impressions) * 100, 2) if total_impressions else 0
        cost_per_conv = round(total_spend / total_conversions, 2) if total_conversions else 0

        # Group summaries
        breakdowns = [breakdown_1, breakdown_2, breakdown_3]
        group_blocks = []
        for b in breakdowns:
            if b and b in df.columns:
                summary = build_group_summary(df, b)
                if summary:
                    group_blocks.append(f"## Breakdown by {b}\n\n{summary}")

        # PROMPT
        prompt = f"""
You are a professional paid media strategist reporting on a DV360 campaign.

Your goal is to deliver a confident, data-driven, first-person performance report. You must sound like a human strategist who understands what the data means — not just repeat it. You are writing a commentary, not giving a speech.

## Executive Summary  
## Performance vs KPIs  
## Breakdown Analysis  
{''.join([f'\n{block}' for block in group_blocks]) if group_blocks else 'No breakdowns selected.'}

## Conversion Analysis  
## Strategic Observations & Recommendations

---

## CAMPAIGN BRIEF
- Objective: {objective}  
- CTR Target: {ctr_target}%  
- CPM Target: SGD {cpm_target}  
- Budget: SGD {budget}  
- Flight: {flight}  
- Primary Metric: {primary_metric}  
- Secondary Metric: {secondary_metric or 'None'}

## OVERALL PERFORMANCE
- Impressions: {total_impressions:,}  
- Clicks: {total_clicks:,}  
- CTR: {ctr:.2f}%  
- Spend: SGD {total_spend:,.2f}  
- CPM: SGD {cpm:,.2f}  
- CPC: SGD {cpc:,.2f}  
- Conversions: {total_conversions:,}  
- Conversion Rate: {conv_rate:.2f}%  
- Cost per Conversion: SGD {cost_per_conv:,.2f}
"""

        print("🧠 Prompt ready. Sending to GPT...")
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            temperature=0.8,
            messages=[
                {"role": "system", "content": "You write structured, strategic campaign insights like a confident media buyer."},
                {"role": "user", "content": prompt}
            ]
        )

        report_text = response.choices[0].message.content
        return JSONResponse(content={"report": report_text})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})


# ---------- CHATBOT ENDPOINT ----------
class InteractionRequest(BaseModel):
    insight: str
    user_prompt: str
    mode: str

@app.post("/interact-insight/")
async def interact_with_insight(request: InteractionRequest):
    try:
        messages = [
            {"role": "system", "content": "You are a paid media analyst. If mode is 'ask', answer the user's question using the insight text. If 'edit', revise the insight accordingly."},
            {"role": "user", "content": f"--- Original Insight ---\n{request.insight}"},
            {"role": "user", "content": f"--- User Prompt ---\n{request.user_prompt}"},
            {"role": "user", "content": f"Mode: {request.mode}"}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.7,
            messages=messages
        )

        result = response.choices[0].message.content
        return JSONResponse(content={"result": result})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})

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

# Initialize FastAPI app
app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- GROUPING SUMMARY HELPER ----------
def build_group_summary(df, group_by_col, label=None):
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
        summary = f"{label or group_by_col}: {row[group_by_col]}\n"
        summary += f"Impressions: {int(row['Impressions'])}, Clicks: {int(row['Clicks'])}, CTR: {row['CTR (%)']:.2f}%\n"
        summary += f"Spend: SGD {row['Spend']:.2f}, CPM: SGD {row['CPM (SGD)']:.2f}, CPC: SGD {row['CPC (SGD)']:.2f}\n"
        summary += f"Conversions: {int(row['Total Conversions'])}, Conversion Rate: {row['Conversion Rate (%)']:.2f}%"
        summaries.append(summary)
    
    return "\n\n".join(summaries)

# ---------- INSIGHT GENERATOR ----------
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
    group_by: str = Form(None)
):
    try:
        contents = await file.read()
        df = pd.read_excel(BytesIO(contents))
        df.columns = df.columns.map(str)

        print("✅ File loaded. Columns:", df.columns.tolist())

        # Check essential columns
        required_columns = ['Impressions', 'Clicks', 'Spend']
        for col in required_columns:
            if col not in df.columns:
                return JSONResponse(status_code=400, content={"error": f"Missing column: {col}"})

        if 'Total Conversions' not in df.columns:
            df['Total Conversions'] = 0

        # Aggregate totals
        total_impressions = df['Impressions'].sum()
        total_clicks = df['Clicks'].sum()
        total_spend = df['Spend'].sum()
        total_conversions = df['Total Conversions'].sum()

        ctr = round((total_clicks / total_impressions) * 100, 2) if total_impressions else 0
        cpm = round((total_spend / total_impressions) * 1000, 2) if total_impressions else 0
        cpc = round(total_spend / total_clicks, 2) if total_clicks else 0
        conv_rate = round((total_conversions / total_impressions) * 100, 2) if total_impressions else 0
        cost_per_conv = round(total_spend / total_conversions, 2) if total_conversions else 0

        # Optional: build summary for selected grouping column
        group_summary = build_group_summary(df, group_by, group_by) if group_by else ""
        group_section = f"## {group_by} PERFORMANCE\n\n{group_summary}" if group_summary else ""

        # Build prompt
        prompt = f"""
You are a professional paid media strategist reporting on a DV360 campaign.

You must deliver a clear, structured, and confident performance commentary using the following format:

## Executive Summary  
## Performance vs KPIs  
## Conversion Analysis  
{f"## {group_by} Performance" if group_summary else ""}
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

{group_section}
"""

        print("🧠 Prompt prepared. Sending to GPT-4-turbo...")

        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            temperature=0.85,
            messages=[
                {"role": "system", "content": "You write strategic, structured campaign insights like a paid media expert."},
                {"role": "user", "content": prompt}
            ]
        )

        result = response.choices[0].message.content
        return JSONResponse(content={"report": result})

    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})


# ---------- COLUMN ANALYSIS ENDPOINT ----------
@app.post("/analyze-columns/")
async def analyze_columns(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_excel(BytesIO(contents))
        df.columns = df.columns.map(str)
        print("📊 Analyzing columns:", df.columns.tolist())

        dimensions = []
        metrics = []

        for col in df.columns:
            unique_vals = df[col].nunique()
            if pd.api.types.is_numeric_dtype(df[col]) and unique_vals > 10:
                metrics.append(col)
            elif unique_vals <= 50:
                dimensions.append(col)

        return JSONResponse(content={"dimensions": dimensions, "metrics": metrics})

    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})


# ---------- CHAT ENDPOINT ----------
class InteractionRequest(BaseModel):
    insight: str
    user_prompt: str
    mode: str

@app.post("/interact-insight/")
async def interact_with_insight(request: InteractionRequest):
    try:
        print("💬 Chat Interaction:", request.dict())
        messages = [
            {"role": "system", "content": "You are a paid media analyst. If mode is 'ask', answer the user's question using the insight text. If mode is 'edit', revise the insight accordingly."},
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
        import traceback
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})

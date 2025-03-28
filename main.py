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

# ========== Setup ==========
load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")
print("🔑 OpenAI Key Loaded?", bool(openai.api_key))

# ========== Helper Functions ==========
def build_group_summary(df, col):
    df[col] = df[col].fillna("Unknown")
    grouped = df.groupby(col).agg({
        'Impressions': 'sum',
        'Clicks': 'sum',
        'Spend': 'sum',
        'Total Conversions': 'sum'
    }).reset_index()

    grouped['CTR (%)'] = (grouped['Clicks'] / grouped['Impressions']) * 100
    grouped['CPM (SGD)'] = (grouped['Spend'] / grouped['Impressions']) * 1000
    grouped['CPC (SGD)'] = grouped['Spend'] / grouped['Clicks']
    grouped['Conversion Rate (%)'] = (grouped['Total Conversions'] / grouped['Impressions']) * 100
    grouped = grouped.fillna(0).sort_values(by='Spend', ascending=False)

    summaries = []
    for _, row in grouped.iterrows():
        if row['Impressions'] < 1000:
            continue
        block = f"{col}: {row[col]}\n"
        block += f"Impressions: {int(row['Impressions'])}, Clicks: {int(row['Clicks'])}, CTR: {row['CTR (%)']:.2f}%\n"
        block += f"Spend: SGD {row['Spend']:.2f}, CPM: SGD {row['CPM (SGD)']:.2f}, CPC: SGD {row['CPC (SGD)']:.2f}\n"
        block += f"Conversions: {int(row['Total Conversions'])}, Conversion Rate: {row['Conversion Rate (%)']:.2f}%"
        summaries.append(block)
    return "\n\n".join(summaries)

# ========== Endpoint ==========
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
    group_columns: str = Form("")  # comma-separated column names
):
    try:
        contents = await file.read()
        df = pd.read_excel(BytesIO(contents))
        print("✅ File loaded. Columns:", df.columns.tolist())

        if 'Total Conversions' not in df.columns:
            df['Total Conversions'] = 0

        required_cols = ['Impressions', 'Clicks', 'Spend']
        for col in required_cols:
            if col not in df.columns:
                return JSONResponse(status_code=400, content={"error": f"Missing column: {col}"})

        df[required_cols + ['Total Conversions']] = df[required_cols + ['Total Conversions']].fillna(0)

        total_impressions = df['Impressions'].sum()
        total_clicks = df['Clicks'].sum()
        total_spend = df['Spend'].sum()
        total_conversions = df['Total Conversions'].sum()

        ctr = round((total_clicks / total_impressions) * 100, 2) if total_impressions else 0
        cpm = round((total_spend / total_impressions) * 1000, 2) if total_impressions else 0
        cpc = round(total_spend / total_clicks, 2) if total_clicks else 0
        conv_rate = round((total_conversions / total_impressions) * 100, 2) if total_impressions else 0
        cost_per_conv = round(total_spend / total_conversions, 2) if total_conversions else 0

        summaries = {}
        for col in group_columns.split(","):
            col = col.strip()
            if col in df.columns:
                summaries[col] = build_group_summary(df, col)

        breakdowns = "\n\n".join([f"### {col}\n{summary}" for col, summary in summaries.items() if summary])

        prompt = f"""
You are a professional paid media strategist reporting on a DV360 display campaign.

You must deliver a clear, structured, and confident performance commentary using the following format:

## Executive Summary  
## Performance vs KPIs  
## Strategic Audience or Line Item Observations  
## Creative & Platform Insights  
## Device or Environment Analysis  
## Conversion Analysis  
## Strategic Observations & Recommendations

Use first-person, natural tone. You must sound analytical and strategic, not like a speech or dashboard. 

Avoid repeating table values without explanation. Always answer “so what?” when citing a number — why did that happen, what does it mean?

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

## BREAKDOWNS

{breakdowns}
"""

        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            temperature=0.8,
            messages=[
                {"role": "system", "content": "You write sharp, confident media campaign insights."},
                {"role": "user", "content": prompt}
            ]
        )

        result = response.choices[0].message.content
        return JSONResponse(content={"report": result})

    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})

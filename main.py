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

load_dotenv()
app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")
print("🔑 Loaded OpenAI API Key?", bool(openai.api_key))

# 1. Analyze uploaded file to extract columns
@app.post("/analyze-columns/")
async def analyze_columns(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_excel(BytesIO(contents))
        df.columns = df.columns.map(str)

        # Identify numeric columns as potential metrics
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        all_cols = df.columns.tolist()
        text_cols = list(set(all_cols) - set(numeric_cols))

        return {
            "dimensions": sorted(text_cols),
            "metrics": sorted(numeric_cols),
            "columns": all_cols
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# 2. Helper to create group summary
def build_group_summary(df, group_by_col):
    summary_text = ""
    if group_by_col in df.columns:
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

        parts = []
        for _, row in grouped.iterrows():
            summary = f"{group_by_col}: {row[group_by_col]}\n"
            summary += f"Impressions: {int(row['Impressions'])}, Clicks: {int(row['Clicks'])}, CTR: {row['CTR (%)']:.2f}%\n"
            summary += f"Spend: SGD {row['Spend']:.2f}, CPM: SGD {row['CPM (SGD)']:.2f}, CPC: SGD {row['CPC (SGD)']:.2f}\n"
            summary += f"Conversions: {int(row['Total Conversions'])}, Conversion Rate: {row['Conversion Rate (%)']:.2f}%"
            parts.append(summary)
        summary_text = "\n\n".join(parts)
    return summary_text

# 3. Main insight endpoint
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

        required = ['Impressions', 'Clicks', 'Spend']
        for col in required:
            if col not in df.columns:
                return JSONResponse(status_code=400, content={"error": f"Missing column: {col}"})

        if 'Total Conversions' not in df.columns:
            df['Total Conversions'] = 0

        total_impressions = df['Impressions'].sum()
        total_clicks = df['Clicks'].sum()
        total_spend = df['Spend'].sum()
        total_conversions = df['Total Conversions'].sum()

        ctr = round((total_clicks / total_impressions) * 100, 2) if total_impressions else 0
        cpm = round((total_spend / total_impressions) * 1000, 2) if total_impressions else 0
        cpc = round(total_spend / total_clicks, 2) if total_clicks else 0
        conv_rate = round((total_conversions / total_impressions) * 100, 2) if total_impressions else 0
        cost_per_conv = round(total_spend / total_conversions, 2) if total_conversions else 0

        # Grouped summary
        group_summary = build_group_summary(df, group_by) if group_by else ""

        # Prompt to GPT
        prompt = f"""
You are a professional paid media strategist reporting on a DV360 campaign.

You must deliver a clear, structured, and confident performance commentary using the following format:

## Executive Summary  
## Performance vs KPIs  
{"## Grouped Breakdown" if group_summary else ""}
## Conversion Analysis  
## Strategic Observations & Recommendations

Be analytical, strategic, and insightful — always try to answer the “so what?” of any performance. Avoid overly generic language.

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

{f"## {group_by} PERFORMANCE\n\n" + group_summary if group_summary else ""}
"""

        print("🧠 Sending prompt to OpenAI...")
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            temperature=0.85,
            messages=[
                {"role": "system", "content": "You write structured, strategic campaign insights like a confident media buyer."},
                {"role": "user", "content": prompt}
            ]
        )

        report_text = response.choices[0].message.content
        return JSONResponse(content={"report": report_text})

    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})

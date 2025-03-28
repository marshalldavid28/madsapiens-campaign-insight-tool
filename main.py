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
import time

# Load environment variables
load_dotenv()
app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")
print("🔑 Loaded OpenAI API Key?", bool(openai.api_key))

# ========== Helpers ==========

def safe_ratio(numerator, denominator):
    return round((numerator / denominator) * 100, 2) if denominator else 0

def build_group_summary(df, group_by_col, label):
    if group_by_col not in df.columns:
        return ""

    if df[group_by_col].nunique() > 25:
        return ""  # too many rows

    if df['Impressions'].sum() < 1000:
        return ""  # too little data

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
        summary = f"{label}: {row[group_by_col]}\n"
        summary += f"Impressions: {int(row['Impressions'])}, Clicks: {int(row['Clicks'])}, CTR: {row['CTR (%)']:.2f}%\n"
        summary += f"Spend: SGD {row['Spend']:.2f}, CPM: SGD {row['CPM (SGD)']:.2f}, CPC: SGD {row['CPC (SGD)']:.2f}\n"
        summary += f"Conversions: {int(row['Total Conversions'])}, Conversion Rate: {row['Conversion Rate (%)']:.2f}%"
        parts.append(summary)
    return "\n\n".join(parts)

# ========== Main Endpoint ==========
@app.post("/generate-insights/")
async def generate_insights(
    file: UploadFile = File(...),
    objective: str = Form(...),
    ctr_target: float = Form(...),
    cpm_target: float = Form(...),
    budget: float = Form(...),
    flight: str = Form(...),
    primary_metric: str = Form(...),
    secondary_metric: str = Form(None)
):
    try:
        t0 = time.time()
        contents = await file.read()
        df = pd.read_excel(BytesIO(contents))
        print("✅ File loaded with columns:", df.columns.tolist())

        required = ['Impressions', 'Clicks', 'Spend']
        for col in required:
            if col not in df.columns:
                return JSONResponse(status_code=400, content={"error": f"Missing column: {col}"})

        if 'Total Conversions' not in df.columns:
            df['Total Conversions'] = 0

        df[required + ['Total Conversions']] = df[required + ['Total Conversions']].fillna(0)

        total_impressions = df['Impressions'].sum()
        total_clicks = df['Clicks'].sum()
        total_spend = df['Spend'].sum()
        total_conversions = df['Total Conversions'].sum()

        ctr = safe_ratio(total_clicks, total_impressions)
        cpm = round((total_spend / total_impressions) * 1000, 2) if total_impressions else 0
        cpc = round(total_spend / total_clicks, 2) if total_clicks else 0
        conv_rate = safe_ratio(total_conversions, total_impressions)
        cost_per_conv = round(total_spend / total_conversions, 2) if total_conversions else 0

        print("✅ Totals calculated")

        # Standard breakdowns
        summaries = {
            "Line Item Breakdown": build_group_summary(df, "Line Item", "Line Item"),
            "Creative Performance": build_group_summary(df, "Creative", "Creative"),
            "Device & OS Analysis": build_group_summary(df, "Device Type", "Device Type") + "\n\n" +
                                    build_group_summary(df, "Device Model", "Device Model")
        }

        # Additional dynamic groupings
        ignored_cols = {'Impressions', 'Clicks', 'Spend', 'Total Conversions',
                        'Line Item', 'Creative', 'Device Type', 'Device Model'}
        dynamic_cols = [col for col in df.columns if col not in ignored_cols and df[col].nunique() <= 25]

        for col in dynamic_cols:
            label = col.replace("_", " ").title()
            summary = build_group_summary(df, col, label)
            if summary:
                summaries[f"{label} Breakdown"] = summary

        # Build prompt
        prompt = f"""
You are a professional paid media strategist reporting on a DV360 display campaign.

Use the following report structure where applicable. Only include a section if relevant data is provided for it:

## Executive Summary  
## Performance vs KPIs  
## Line Item Breakdown  
## Creative Performance  
## Device & OS Analysis  
## Conversion Analysis  
## Strategic Observations & Recommendations

Keep your tone professional and data-literate. Use the data provided to explain *why* something performed the way it did. Avoid repeating metric names—interpret them.

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

        for title, section in summaries.items():
            if section.strip():
                prompt += f"\n\n## {title}\n{section.strip()}"

        print("📏 Prompt length:", len(prompt.split()))
        print("⏳ Sending to OpenAI")

        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            temperature=0.85,
            messages=[
                {"role": "system", "content": "You write structured, strategic campaign insights like a confident media buyer."},
                {"role": "user", "content": prompt}
            ]
        )

        report_text = response.choices[0].message.content
        print("✅ Report generated in", round(time.time() - t0, 2), "seconds")
        return JSONResponse(content={"report": report_text})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})


# ========== Chat Endpoint ==========
class InteractionRequest(BaseModel):
    insight: str
    user_prompt: str
    mode: str

@app.post("/interact-insight/")
async def interact_with_insight(request: InteractionRequest):
    try:
        print("📥 Chat interaction:", request.dict())

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

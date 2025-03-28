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
import traceback

# Load environment variables
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

# OpenAI key
openai.api_key = os.getenv("OPENAI_API_KEY")
print("🔑 Loaded OpenAI API Key?", bool(openai.api_key))

# Column alias mapping
COLUMN_ALIASES = {
    "Spend": ["Spend", "Spends", "Media Cost", "Cost", "Total Spend"],
    "Clicks": ["Clicks", "Click", "Click Count", "Total Clicks"],
    "Impressions": ["Impressions", "Imps", "Total Impressions"],
    "Total Conversions": ["Total Conversions", "Conversions", "All Conversions", "Post Click Conversions"]
}

def resolve_column_aliases(df, required_fields, alias_map):
    resolved = {}
    for key in required_fields:
        aliases = alias_map.get(key, [])
        for alias in aliases:
            if alias in df.columns:
                resolved[key] = alias
                break
        else:
            if key == "Total Conversions":
                df[key] = 0
                resolved[key] = key
            else:
                raise ValueError(f"Missing column: {key} (looked for: {aliases})")
    return resolved

def build_group_summary(df, group_by_col, label, col_map):
    summary_text = ""
    if group_by_col in df.columns:
        grouped = df.groupby(group_by_col).agg({
            col_map['Impressions']: 'sum',
            col_map['Clicks']: 'sum',
            col_map['Spend']: 'sum',
            col_map['Total Conversions']: 'sum'
        }).reset_index()

        grouped['CTR (%)'] = (grouped[col_map['Clicks']] / grouped[col_map['Impressions']]) * 100
        grouped['CPM (SGD)'] = (grouped[col_map['Spend']] / grouped[col_map['Impressions']]) * 1000
        grouped['CPC (SGD)'] = grouped[col_map['Spend']] / grouped[col_map['Clicks']]
        grouped['Conversion Rate (%)'] = (grouped[col_map['Total Conversions']] / grouped[col_map['Impressions']]) * 100
        grouped = grouped.fillna(0).sort_values(by=col_map['Spend'], ascending=False).head(5)

        parts = []
        for _, row in grouped.iterrows():
            summary = f"{label}: {row[group_by_col]}\n"
            summary += f"Impressions: {int(row[col_map['Impressions']])}, Clicks: {int(row[col_map['Clicks']])}, CTR: {row['CTR (%)']:.2f}%\n"
            summary += f"Spend: SGD {row[col_map['Spend']]:.2f}, CPM: SGD {row['CPM (SGD)']:.2f}, CPC: SGD {row['CPC (SGD)']:.2f}\n"
            summary += f"Conversions: {int(row[col_map['Total Conversions']])}, Conversion Rate: {row['Conversion Rate (%)']:.2f}%"
            parts.append(summary)
        summary_text = "\n\n".join(parts)
    return summary_text

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
        contents = await file.read()
        df = pd.read_excel(BytesIO(contents))
        print("✅ Excel file loaded with columns:", df.columns.tolist())

        required_fields = ["Spend", "Clicks", "Impressions", "Total Conversions"]
        col_map = resolve_column_aliases(df, required_fields, COLUMN_ALIASES)

        total_impressions = df[col_map['Impressions']].sum()
        total_clicks = df[col_map['Clicks']].sum()
        total_spend = df[col_map['Spend']].sum()
        total_conversions = df[col_map['Total Conversions']].sum()

        ctr = round((total_clicks / total_impressions) * 100, 2) if total_impressions else 0
        cpm = round((total_spend / total_impressions) * 1000, 2) if total_impressions else 0
        cpc = round(total_spend / total_clicks, 2) if total_clicks else 0
        conv_rate = round((total_conversions / total_impressions) * 100, 2) if total_impressions else 0
        cost_per_conv = round(total_spend / total_conversions, 2) if total_conversions else 0

        line_item_summary = build_group_summary(df, "Line Item", "Line Item", col_map)
        creative_summary = build_group_summary(df, "Creative", "Creative", col_map)
        device_summary = build_group_summary(df, "Device Type", "Device Type", col_map)
        os_summary = build_group_summary(df, "Device Model", "Device Model", col_map)

        prompt = f"""
You are a professional paid media strategist reporting on a DV360 campaign.

You must deliver a clear, structured, and confident performance commentary using the following format:

## Executive Summary  
## Performance vs KPIs  
## Line Item Breakdown  
## Creative Performance  
## Device & OS Analysis  
## Conversion Analysis  
## Strategic Observations & Recommendations

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

## LINE ITEM PERFORMANCE
{line_item_summary or 'No line item data available.'}

## CREATIVE PERFORMANCE
{creative_summary or 'No creative data available.'}

## DEVICE PERFORMANCE
{device_summary or 'No device data available.'}

## OS PERFORMANCE
{os_summary or 'No OS data available.'}

## STRATEGIC OBSERVATIONS
Please conclude with future suggestions, recommendations and next steps.
"""

        print("🧠 Prompt ready. Sending to GPT...")
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
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})

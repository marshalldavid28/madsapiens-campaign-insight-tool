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


# ========== Helpers ==========
def build_group_summary(df, group_by_col, label):
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
            parts.append(
                f"{label}: {row[group_by_col]}\n"
                f"Impressions: {int(row['Impressions'])}, Clicks: {int(row['Clicks'])}, CTR: {row['CTR (%)']:.2f}%\n"
                f"Spend: SGD {row['Spend']:.2f}, CPM: SGD {row['CPM (SGD)']:.2f}, CPC: SGD {row['CPC (SGD)']:.2f}\n"
                f"Conversions: {int(row['Total Conversions'])}, Conversion Rate: {row['Conversion Rate (%)']:.2f}%"
            )
        summary_text = "\n\n".join(parts)
    return summary_text


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
    secondary_metric: str = Form(None),
    breakdown_1: str = Form(None),
    breakdown_2: str = Form(None),
    breakdown_3: str = Form(None)
):
    try:
        contents = await file.read()
        df = pd.read_excel(BytesIO(contents))
        print("✅ Loaded Excel file with columns:", df.columns.tolist())

        df.columns = df.columns.str.strip()

        # Alias handling
        col_map = {
            'spend': 'Spend',
            'clicks': 'Clicks',
            'impressions': 'Impressions',
            'conversions': 'Total Conversions'
        }
        df.rename(columns={col: std for col, std in col_map.items() if col in df.columns.str.lower()}, inplace=True)

        # Check minimum required columns
        required = ['Impressions', 'Clicks', 'Spend']
        for col in required:
            if col not in df.columns:
                return JSONResponse(status_code=400, content={"error": f"Missing column: {col}"})

        if 'Total Conversions' not in df.columns:
            df['Total Conversions'] = 0

        # Metric calculations
        total_impressions = df['Impressions'].sum()
        total_clicks = df['Clicks'].sum()
        total_spend = df['Spend'].sum()
        total_conversions = df['Total Conversions'].sum()

        ctr = round((total_clicks / total_impressions) * 100, 2) if total_impressions else 0
        cpm = round((total_spend / total_impressions) * 1000, 2) if total_impressions else 0
        cpc = round(total_spend / total_clicks, 2) if total_clicks else 0
        conv_rate = round((total_conversions / total_impressions) * 100, 2) if total_impressions else 0
        cost_per_conv = round(total_spend / total_conversions, 2) if total_conversions else 0

        # Dynamic breakdowns
        breakdowns = [b for b in [breakdown_1, breakdown_2, breakdown_3] if b]
        summaries = []
        for breakdown in breakdowns:
            if breakdown in df.columns:
                summaries.append(build_group_summary(df, breakdown, breakdown))

        breakdown_text = "\n\n".join([f"### {breakdown}\n{text}" for breakdown, text in zip(breakdowns, summaries) if text])

        # Final prompt
        prompt = (
            f"You are a professional paid media strategist reporting on a DV360 Display campaign.\n\n"
            f"Your tone is confident, data-driven, and insightful. Use the following structure:\n\n"
            f"## Executive Summary\n"
            f"## Performance vs KPIs\n"
            f"## Conversion Analysis\n"
            f"{''.join([f'## {b} Analysis\n' for b in breakdowns])}"
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
        import traceback
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})

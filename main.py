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

# CORS
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

# -------- Helpers --------

def clean_sheet(df):
    if df.columns.str.contains("Unnamed").sum() > len(df.columns) // 2:
        df.columns = df.iloc[2]
        df = df.drop(index=[0, 1, 2]).reset_index(drop=True)
    return df

def group_summary(df, group_by_col):
    if group_by_col not in df.columns:
        return ""
    df[group_by_col] = df[group_by_col].fillna("Unknown")
    numeric_cols = ["Impressions", "Clicks", "Spend", "Total Conversions"]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0
    df[numeric_cols] = df[numeric_cols].fillna(0)
    grouped = df.groupby(group_by_col).agg("sum").reset_index()
    grouped["CTR (%)"] = (grouped["Clicks"] / grouped["Impressions"]) * 100
    grouped["CPM (SGD)"] = (grouped["Spend"] / grouped["Impressions"]) * 1000
    grouped["CPC (SGD)"] = grouped["Spend"] / grouped["Clicks"]
    grouped["Conversion Rate (%)"] = (grouped["Total Conversions"] / grouped["Impressions"]) * 100
    grouped = grouped.fillna(0).sort_values(by="Spend", ascending=False).head(5)

    parts = []
    for _, row in grouped.iterrows():
        summary = f"{group_by_col}: {row[group_by_col]}\n"
        summary += f"Impressions: {int(row['Impressions'])}, Clicks: {int(row['Clicks'])}, CTR: {row['CTR (%)']:.2f}%\n"
        summary += f"Spend: SGD {row['Spend']:.2f}, CPM: SGD {row['CPM (SGD)']:.2f}, CPC: SGD {row['CPC (SGD)']:.2f}\n"
        summary += f"Conversions: {int(row['Total Conversions'])}, Conversion Rate: {row['Conversion Rate (%)']:.2f}%"
        parts.append(summary)
    return "\n\n".join(parts)

# -------- Insights Endpoint --------

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
        xls = pd.ExcelFile(BytesIO(contents))
        summaries = {}
        base_df = None

        for sheet in xls.sheet_names:
            df = clean_sheet(xls.parse(sheet))
            df.columns = df.columns.map(str)
            if sheet.lower() in ["region", "audiences", "device type", "device", "country", "city", "os"]:
                summaries[sheet] = group_summary(df, sheet if sheet in df.columns else df.columns[0])
            elif base_df is None and {"Impressions", "Clicks", "Spend", "Total Conversions"}.issubset(df.columns):
                base_df = df

        if base_df is None:
            return JSONResponse(status_code=400, content={"error": "No base data found with required columns."})

        total_impressions = base_df["Impressions"].sum()
        total_clicks = base_df["Clicks"].sum()
        total_spend = base_df["Spend"].sum()
        total_conversions = base_df["Total Conversions"].sum()

        ctr = round((total_clicks / total_impressions) * 100, 2) if total_impressions else 0
        cpm = round((total_spend / total_impressions) * 1000, 2) if total_impressions else 0
        cpc = round(total_spend / total_clicks, 2) if total_clicks else 0
        conv_rate = round((total_conversions / total_impressions) * 100, 2) if total_impressions else 0
        cost_per_conv = round(total_spend / total_conversions, 2) if total_conversions else 0

        prompt = f"""You are a media strategist reporting on a DV360 campaign.

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

## ADDITIONAL BREAKDOWNS
""" + "\n\n".join([f"### {k}\n{v}" for k, v in summaries.items() if v])

        response = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=0.85,
            messages=[
                {"role": "system", "content": "You write strategic, confident campaign insights."},
                {"role": "user", "content": prompt}
            ]
        )

        report_text = response.choices[0].message.content
        return JSONResponse(content={"report": report_text})

    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})

# -------- Chat Endpoint --------

class InteractionRequest(BaseModel):
    insight: str
    user_prompt: str
    mode: str

@app.post("/interact-insight/")
async def interact_with_insight(request: InteractionRequest):
    try:
        print("📥 Chat Request Received:", request.dict())
        print(f"Mode: {request.mode}\nPrompt: {request.user_prompt[:150]}...")

        instruction = (
            "You are a strategic paid media analyst. "
            "If mode is 'ask', answer the user’s question based only on the insight. "
            "If mode is 'edit', revise the insight based on the user's instruction."
        )

        messages = [
            {"role": "system", "content": instruction},
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

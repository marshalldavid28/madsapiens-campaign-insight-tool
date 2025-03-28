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

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")
print("🔑 Loaded OpenAI API Key?", bool(openai.api_key))

# -----------------------------
# Utility: Detect Metrics & Dimensions
# -----------------------------
def detect_metrics_and_dimensions(df):
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    dimension_cols = [col for col in df.columns if col not in numeric_cols]
    metrics = [col for col in numeric_cols if df[col].sum() > 0]
    return dimension_cols, metrics

@app.post("/analyze-columns/")
async def analyze_columns(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_excel(BytesIO(contents))
        dimensions, metrics = detect_metrics_and_dimensions(df)
        return JSONResponse(content={
            "dimensions": dimensions,
            "metrics": metrics
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# -----------------------------
# Utility: Group Summary Builder
# -----------------------------
def build_group_summary(df, group_col, metrics):
    if group_col not in df.columns:
        return ""
    if "Impressions" not in df.columns or df["Impressions"].sum() < 1000:
        return ""

    summary = []
    group_df = df.groupby(group_col).agg({metric: 'sum' for metric in metrics}).reset_index()

    # Add calculated metrics
    if "Clicks" in metrics and "Impressions" in metrics:
        group_df["CTR (%)"] = (group_df["Clicks"] / group_df["Impressions"]) * 100
    if "Spend" in metrics and "Impressions" in metrics:
        group_df["CPM (SGD)"] = (group_df["Spend"] / group_df["Impressions"]) * 1000
    if "Spend" in metrics and "Clicks" in metrics:
        group_df["CPC (SGD)"] = group_df["Spend"] / group_df["Clicks"]
    if "Total Conversions" in metrics and "Impressions" in metrics:
        group_df["Conversion Rate (%)"] = (group_df["Total Conversions"] / group_df["Impressions"]) * 100

    group_df = group_df.fillna(0).sort_values(by='Impressions', ascending=False).head(5)

    for _, row in group_df.iterrows():
        block = f"{group_col}: {row[group_col]}"
        for metric in group_df.columns[1:]:
            val = row[metric]
            if isinstance(val, float):
                block += f"\n{metric}: {val:.2f}"
            else:
                block += f"\n{metric}: {val}"
        summary.append(block)
    return "\n\n".join(summary)

# -----------------------------
# Generate Insights Endpoint
# -----------------------------
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
    breakdowns: str = Form("")
):
    try:
        start_time = time.time()
        contents = await file.read()
        df = pd.read_excel(BytesIO(contents))
        print("✅ File loaded with shape:", df.shape)

        breakdown_cols = [col.strip() for col in breakdowns.split(",") if col.strip()]
        print("📊 Breakdown columns selected:", breakdown_cols)

        for col in ['Impressions', 'Clicks', 'Spend']:
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

        # Summaries
        summaries = {}
        metrics = ['Impressions', 'Clicks', 'Spend', 'Total Conversions']
        for col in breakdown_cols:
            summaries[col] = build_group_summary(df, col, metrics)

        summary_blocks = "\n\n".join(
            [f"## {col} Breakdown\n{txt}" for col, txt in summaries.items() if txt]
        )

        prompt = f"""
You are a strategic media analyst reporting on a DV360 Display campaign.

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

{summary_blocks}

## STRATEGIC OBSERVATIONS & RECOMMENDATIONS
Please conclude with a strategic section based on the trends you noticed.
""".strip()

        print("🧠 Prompt size:", len(prompt), "chars")
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            temperature=0.85,
            messages=[
                {"role": "system", "content": "You write structured, strategic campaign insights like a confident media buyer."},
                {"role": "user", "content": prompt}
            ]
        )
        print("✅ GPT response received in", round(time.time() - start_time, 2), "s")
        return JSONResponse(content={"report": response.choices[0].message.content})

    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})

# -----------------------------
# Chat Endpoint
# -----------------------------
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
        import traceback
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import pandas as pd
import openai
import os
import time
from io import BytesIO
import traceback

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

print("🔐 OpenAI Key loaded?", bool(openai.api_key))

# ========== Analyze Columns Endpoint ==========
@app.post("/analyze-columns/")
async def analyze_columns(file: UploadFile = File(...)):
    try:
        print("📥 /analyze-columns/ called...")
        contents = await file.read()
        df = pd.read_excel(BytesIO(contents))
        df.columns = df.columns.map(str)
        print("🧾 Columns in uploaded file:", df.columns.tolist())

        dimensions = []
        for col in df.columns:
            unique_vals = df[col].nunique()
            if df[col].dtype == "object" or unique_vals < 25:
                dimensions.append(col)

        print("✅ Dimensions selected for breakdown:", dimensions)
        return JSONResponse(content={"dimensions": dimensions})

    except Exception as e:
        print("❌ Error in /analyze-columns/:", traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})


# ========== Generate Insights Endpoint ==========
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
        t0 = time.time()
        contents = await file.read()
        df = pd.read_excel(BytesIO(contents))
        df.columns = df.columns.map(str)
        print("📊 Uploaded Data Columns:", df.columns.tolist())

        required_metrics = ['Impressions', 'Clicks', 'Spend']
        for col in required_metrics:
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

        # Group breakdowns based on user selection
        breakdowns_selected = [b.strip() for b in breakdowns.split(",") if b.strip()]
        summaries = []
        for col in breakdowns_selected:
            if col in df.columns:
                group = df.groupby(col).agg({
                    'Impressions': 'sum',
                    'Clicks': 'sum',
                    'Spend': 'sum',
                    'Total Conversions': 'sum'
                }).reset_index()

                group['CTR (%)'] = (group['Clicks'] / group['Impressions']) * 100
                group['CPM (SGD)'] = (group['Spend'] / group['Impressions']) * 1000
                group['CPC (SGD)'] = group['Spend'] / group['Clicks']
                group['Conversion Rate (%)'] = (group['Total Conversions'] / group['Impressions']) * 100
                group = group.fillna(0).sort_values(by='Spend', ascending=False).head(5)

                for _, row in group.iterrows():
                    summary = f"{col}: {row[col]}\n"
                    summary += f"Impressions: {int(row['Impressions'])}, Clicks: {int(row['Clicks'])}, CTR: {row['CTR (%)']:.2f}%\n"
                    summary += f"Spend: SGD {row['Spend']:.2f}, CPM: SGD {row['CPM (SGD)']:.2f}, CPC: SGD {row['CPC (SGD)']:.2f}\n"
                    summary += f"Conversions: {int(row['Total Conversions'])}, Conversion Rate: {row['Conversion Rate (%)']:.2f}%\n"
                    summaries.append(summary)
            else:
                print(f"⚠️ Skipping missing column: {col}")

        summary_text = "\n\n".join(summaries) if summaries else "No breakdowns available or selected."

        # Prompt
        prompt = f"""You are a professional paid media strategist writing a campaign performance report.

## Campaign Brief
- Objective: {objective}
- CTR Target: {ctr_target}%
- CPM Target: SGD {cpm_target}
- Budget: SGD {budget}
- Flight: {flight}
- Primary Metric: {primary_metric}
- Secondary Metric: {secondary_metric or 'None'}

## Overall Performance
- Impressions: {total_impressions:,}
- Clicks: {total_clicks:,}
- CTR: {ctr:.2f}%
- Spend: SGD {total_spend:,.2f}
- CPM: SGD {cpm:,.2f}
- CPC: SGD {cpc:,.2f}
- Conversions: {total_conversions:,}
- Conversion Rate: {conv_rate:.2f}%
- Cost per Conversion: SGD {cost_per_conv:,.2f}

## Grouped Insights
{summary_text}

## Strategic Recommendations
Write closing thoughts about what worked, what didn’t, and what to optimize next time.
"""

        print("📤 Sending prompt to GPT...")
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            temperature=0.85,
            messages=[
                {"role": "system", "content": "You write strategic, human-like campaign performance insights."},
                {"role": "user", "content": prompt}
            ]
        )

        result = response.choices[0].message.content
        print("✅ Insight generated in", round(time.time() - t0, 2), "seconds")
        return JSONResponse(content={"report": result})

    except Exception as e:
        print("❌ Error in /generate-insights/:", traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})


# ========== Chat Interaction ==========
class InteractionRequest(BaseModel):
    insight: str
    user_prompt: str
    mode: str

@app.post("/interact-insight/")
async def interact_with_insight(request: InteractionRequest):
    try:
        messages = [
            {"role": "system", "content": "You are a paid media analyst. Answer questions or improve the report as needed."},
            {"role": "user", "content": f"--- Insight ---\n{request.insight}"},
            {"role": "user", "content": f"--- Request ---\n{request.user_prompt}"},
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
        print("❌ Chat Error:", traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})

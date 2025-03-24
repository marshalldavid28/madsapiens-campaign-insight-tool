from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import openai
import os
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Allow CORS for local dev or frontend usage
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.post("/generate-insights/")
async def generate_insights(
    file: UploadFile = File(...),
    objective: str = Form(...),
    ctr_target: float = Form(...),
    cpm_target: float = Form(...),
    budget: float = Form(...),
    flight: str = Form(...)
):
    try:
        contents = await file.read()
        df = pd.read_excel(BytesIO(contents))

        df['Spend'] = df['Spend'].str.replace('S$', '', regex=False).astype(float)

        total_impressions = df['Impressions'].sum()
        total_clicks = df['Clicks'].sum()
        total_spend = df['Spend'].sum()
        total_conversions = df['Total Conversions'].sum()

        ctr = round((total_clicks / total_impressions) * 100, 2)
        cpm = round((total_spend / total_impressions) * 1000, 2)
        cpc = round(total_spend / total_clicks, 2)
        conv_rate = round((total_conversions / total_clicks) * 100, 2) if total_clicks else 0
        cost_per_conv = round(total_spend / total_conversions, 2) if total_conversions else 0

        prompt = f"""
You are a digital strategist who personally ran a DV360 campaign.

Write a professional, confident, first-person report summarizing campaign performance.
Be detailed and structured. Use a clear sectioned format like this:

1. Executive Summary
2. Platform Performance vs KPIs
3. Line Item Observations
4. Conversion Analysis
5. Recommendations / Strategy Updates

Use the data below. Where helpful, feel free to explain performance trends logically, based only on available metrics. Do not guess optimizations — but you may add two or three possible contributing factors (clearly marked).

Data:
- Objective: {objective}
- Budget: SGD {budget}
- Flight: {flight}
- CTR Target: {ctr_target}%
- CPM Target: SGD {cpm_target}

Performance:
- Impressions: {total_impressions}
- Clicks: {total_clicks}
- CTR: {ctr}%
- Spend: SGD {total_spend}
- CPM: SGD {cpm}
- CPC: SGD {cpc}
- Conversions (Landing Page Visits): {total_conversions}
- Conversion Rate: {conv_rate}%
- Cost per Conversion: SGD {cost_per_conv}

Write in first person, like a strategist explaining this to a team or client. Be sharp, confident, and clear. Keep it realistic — not too fluffy. Expand on the insights where useful.
"""

        response = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=0.8,
            messages=[
                {"role": "system", "content": "You are a professional digital strategist who writes structured, confident campaign performance reports in first person."},
                {"role": "user", "content": prompt}
            ]
        )

        report_text = response.choices[0].message.content
        return JSONResponse(content={"report": report_text})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

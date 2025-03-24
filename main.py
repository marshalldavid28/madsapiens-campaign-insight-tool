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

        # Clean Spend column
        df['Spend'] = df['Spend'].str.replace('S$', '', regex=False).astype(float)

        # Basic metrics
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
You are a digital media strategist who ran a DV360 campaign.
Generate a structured insights report in a professional but human tone.
The user will manually insert optimization notes later — include clear placeholders for those.

CAMPAIGN INFO:
- Objective: {objective}
- Budget: SGD {budget}
- Flight Dates: {flight}
- CTR Target: {ctr_target}%
- CPM Target: SGD {cpm_target}

PERFORMANCE:
- Impressions: {total_impressions}
- Clicks: {total_clicks}
- CTR: {ctr}%
- Spend: SGD {total_spend}
- CPM: SGD {cpm}
- CPC: SGD {cpc}
- Conversions: {total_conversions}
- Conversion Rate: {conv_rate}%
- Cost/Conversion: SGD {cost_per_conv}

Start with a short Executive Summary.
Then break the report into structured sections:
1. Platform-Level Performance vs KPIs
2. Line Item-Level Observations (placeholder: [insert user insights])
3. Conversion Analysis
4. Recommendations (placeholder: [insert user suggestions])

Avoid guessing what optimizations were done — just mark them clearly.
"""

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You write reports like a digital strategist."},
                {"role": "user", "content": prompt}
            ]
        )

        report_text = response.choices[0].message.content
        return JSONResponse(content={"report": report_text})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

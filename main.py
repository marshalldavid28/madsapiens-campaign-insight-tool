from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import openai
from io import BytesIO
import os
from dotenv import load_dotenv

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
        if len(contents) == 0:
            return JSONResponse(status_code=400, content={"error": "Uploaded file is empty."})

        df = pd.read_excel(BytesIO(contents))

        required_columns = ['Impressions', 'Clicks', 'Spend', 'Total Conversions']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Dataset is missing required columns.",
                    "missing_columns": missing_columns,
                    "available_columns": df.columns.tolist()
                }
            )

        if df['Spend'].dtype == 'object':
            df['Spend'] = df['Spend'].str.replace('S$', '', regex=False).astype(float)

        total_impressions = df['Impressions'].sum()
        total_clicks = df['Clicks'].sum()
        total_spend = df['Spend'].sum()
        total_conversions = df['Total Conversions'].sum()

        ctr = round((total_clicks / total_impressions) * 100, 2) if total_impressions else 0
        cpm = round((total_spend / total_impressions) * 1000, 2) if total_impressions else 0
        cpc = round(total_spend / total_clicks, 2) if total_clicks else 0
        conv_rate = round((total_conversions / total_impressions) * 100, 2) if total_impressions else 0
        cost_per_conv = round(total_spend / total_conversions, 2) if total_conversions else 0

        line_item_summary = ""
        if 'Insertion Order' in df.columns and 'Line Item' in df.columns:
            grouped = df.groupby(['Insertion Order', 'Line Item']).agg({
                'Impressions': 'sum',
                'Clicks': 'sum',
                'Spend': 'sum',
                'Total Conversions': 'sum'
            }).reset_index()

            grouped['CTR (%)'] = (grouped['Clicks'] / grouped['Impressions']) * 100
            grouped['CPM (SGD)'] = (grouped['Spend'] / grouped['Impressions']) * 1000
            grouped['CPC (SGD)'] = grouped['Spend'] / grouped['Clicks']
            grouped['Conversion Rate (%)'] = (grouped['Total Conversions'] / grouped['Impressions']) * 100
            grouped = grouped.fillna(0)
            grouped = grouped.sort_values(by='Spend', ascending=False)

            summaries = []
            for _, row in grouped.iterrows():
                item = f"Line Item: {row['Line Item']} (IO: {row['Insertion Order']})\n"
                item += f"Impressions: {int(row['Impressions'])}, Clicks: {int(row['Clicks'])}, CTR: {row['CTR (%)']:.2f}%\n"
                item += f"Spend: SGD {row['Spend']:.2f}, CPM: SGD {row['CPM (SGD)']:.2f}, CPC: SGD {row['CPC (SGD)']:.2f}\n"
                item += f"Conversions: {int(row['Total Conversions'])}, Conversion Rate: {row['Conversion Rate (%)']:.2f}%"
                summaries.append(item)

            line_item_summary = "\n\n".join(summaries)

        prompt = f"""
You are a professional paid media strategist reporting on a DV360 campaign.

Your goal is to deliver a confident, data-driven, first-person performance report.  
Use the following structure:

1. Executive Summary  
2. Performance vs KPIs  
3. Line Item Breakdown  
4. Conversion Analysis  
5. Strategic Observations or Next Steps

--- CAMPAIGN BRIEF ---
- Objective: {objective}
- CTR Target: {ctr_target}%
- CPM Target: SGD {cpm_target}
- Budget: SGD {budget}
- Flight: {flight}
- Primary Metric: {primary_metric}
- Secondary Metric: {secondary_metric or 'None'}

Focus your analysis on the Primary Metric. Only refer to the Secondary Metric where it adds value. 
Avoid diving into unrelated metrics unless critical.

--- OVERALL PERFORMANCE ---
- Impressions: {total_impressions:,}
- Clicks: {total_clicks:,}
- CTR: {ctr:.2f}%
- Spend: SGD {total_spend:,.2f}
- CPM: SGD {cpm:,.2f}
- CPC: SGD {cpc:,.2f}
- Conversions: {total_conversions:,}
- Conversion Rate: {conv_rate:.2f}%
- Cost per Conversion: SGD {cost_per_conv:,.2f}

--- LINE ITEM PERFORMANCE ---
{line_item_summary}

Write in a confident first-person voice.
"""
        print("\n🔍 Final Prompt Sent to GPT:\n")
        print(prompt)

        response = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=0.85,
            messages=[
                {"role": "system", "content": "You write structured, strategic campaign insights like a confident media buyer."},
                {"role": "user", "content": prompt}
            ]
        )

        report_text = response.choices[0].message.content if response.choices else "No response generated."
        return JSONResponse(content={"report": report_text})

    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})


class InteractionRequest(BaseModel):
    insight: str
    user_prompt: str
    mode: str  # "ask" or "edit"

@app.post("/interact-insight/")
async def interact_with_insight(request: InteractionRequest):
    try:
        instruction = (
            "You are a paid media analyst. "
            "If mode is 'ask', answer the user’s question using the insight text only. "
            "If mode is 'edit', revise or improve the insight based on the user's prompt."
        )

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": f"--- Original Insight ---\n{request.insight}"},
            {"role": "user", "content": f"--- User Prompt ---\n{request.user_prompt}"},
            {"role": "user", "content": f"Mode: {request.mode}"}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=0.85,
            messages=messages
        )

        result = response.choices[0].message.content
        return JSONResponse(content={"result": result})

    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})

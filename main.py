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

# OpenAI key
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
            summary = f"{label}: {row[group_by_col]}\n"
            summary += f"Impressions: {int(row['Impressions'])}, Clicks: {int(row['Clicks'])}, CTR: {row['CTR (%)']:.2f}%\n"
            summary += f"Spend: SGD {row['Spend']:.2f}, CPM: SGD {row['CPM (SGD)']:.2f}, CPC: SGD {row['CPC (SGD)']:.2f}\n"
            summary += f"Conversions: {int(row['Total Conversions'])}, Conversion Rate: {row['Conversion Rate (%)']:.2f}%"
            parts.append(summary)
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
    secondary_metric: str = Form(None)
):
    try:
        contents = await file.read()
        df = pd.read_excel(BytesIO(contents))
        print("✅ Excel file loaded with columns:", df.columns.tolist())

        required_columns = ['Impressions', 'Clicks', 'Spend']
        for col in required_columns:
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

        # Group summaries
        line_item_summary = build_group_summary(df, "Line Item", "Line Item")
        creative_summary = build_group_summary(df, "Creative", "Creative")
        device_summary = build_group_summary(df, "Device Type", "Device Type")
        os_summary = build_group_summary(df, "Device Model", "Device Model")

        # Prompt to GPT
        prompt = f"""
You are a professional paid media strategist reporting on a DV360 campaign.

Your goal is to deliver a confident, data-driven, first-person performance report. You must sound like a human strategist who understands what the data means — not just repeat it. You are writing a commentary, not giving a speech.

You must use the following report structure and write each section with the exact same headings:

## Executive Summary  
## Performance vs KPIs  
## Line Item Breakdown  
## Creative Performance  
## Device & OS Analysis  
## Conversion Analysis  
## Strategic Observations & Recommendations

Some notes for you to keep in mind about best practices of writing insights:

1. Use clear and concise language: Avoid using technical jargon or complex marketing terminology that may confuse your clients. Instead, use simple, easy-to-understand language to explain your findings.
2. Focus on key metrics: Identify the most important metrics that matter to your clients and focus on those in your reports. This will help them quickly grasp the value of their online campaigns.
3. Focus on achievements: Highlight the achievements and successes of your clients’ online campaigns, rather than just reporting on metrics.
4. Be transparent about challenges: If your clients’ online campaigns are not performing as expected, be transparent about the challenges and provide recommendations for improvement.
5. When you analyse the line items provided to you, read the name of the line items and try to understand what audience segment is being targeted. Use that to form your insights instead of writing out the whole line item name all the time, which can be long and hard to read.
6. Campaign analytics provides granular insights into audience behavior, preferences, and engagement. 
7. By understanding which segments respond best to certain messages or channels, marketers can tailor their campaigns with precision, ensuring that the right message reaches the right audience at the optimal time.
8. With finite resources, it's essential to ensure that every marketing dollar is well-spent. Insights should help identify high-performing campaigns and those that might need reevaluation. 
9. This ensures that marketing budgets are allocated to campaigns that deliver the best results
10. While you do need to follow the primary and secondary metrics and the logic given to you below in this prompt, I dont want you to mention them as primary metric and secondary metric in your insights. Be natural about it.
11. Dont use sentences like "I am proud to announce" or "Im pleases to say that" - This should not be like a speech. It's a commentary, written professionally. 
12. Avoid the approach where data is explained but no reasoning is given - always try and offer up some reason as to why something happened. The question of "so what?" should be answered. Audience A performed with a highest CTR - So what? Try and offer explanations in that regrd wherever possible. Dont force-fit, but try and look for aveneues to fill in that gap wherever relevant.

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
        import traceback
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
        import traceback
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})

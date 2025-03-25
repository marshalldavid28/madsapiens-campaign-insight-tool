from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
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
    flight: str = Form(...)
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
        conv_rate = round((total_conversions / total_clicks) * 100, 2) if total_clicks else 0
        cost_per_conv = round(total_spend / total_conversions, 2) if total_conversions else 0

        # Format line item summary for GPT (flush markdown format)
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
            grouped['Conversion Rate (%)'] = (grouped['Total Conversions'] / grouped['Clicks']) * 100
            grouped = grouped.fillna(0)
            grouped = grouped.sort_values(by='Spend', ascending=False)  # Sort by spend to prioritize top lines

            summaries = []
            for _, row in grouped.iterrows():
                item = f"- **Line Item**: {row['Line Item']} (IO: {row['Insertion Order']})\n"
                item += f"  - Impressions: {int(row['Impressions'])}\n"
                item += f"  - Clicks: {int(row['Clicks'])}\n"
                item += f"  - CTR: {row['CTR (%)']:.2f}%\n"
                item += f"  - Spend: SGD {row['Spend']:.2f}\n"
                item += f"  - CPM: SGD {row['CPM (SGD)']:.2f}\n"
                item += f"  - CPC: SGD {row['CPC (SGD)']:.2f}\n"
                item += f"  - Conversions: {int(row['Total Conversions'])}\n"
                item += f"  - Conversion Rate: {row['Conversion Rate (%)']:.2f}%"
                summaries.append(item)

            line_item_summary = "\n\n" + "\n\n".join(summaries[:10]) + "\n\n--- END LINE ITEM DATA ---"  # Add header/footer markers

        prompt = f"""
You are a professional paid media strategist reporting on a DV360 campaign.

Write a clear, structured, confident report in first person, using the following outline:

1. Executive Summary (top-line outcomes)
2. Performance vs KPIs (compare against CTR/CPM targets)
3. Line Item Breakdown (analyze and interpret the real-world data below)
4. Conversion Analysis
5. Strategic Observations or Next Steps (only using the data provided)

CAMPAIGN BRIEF:
- Objective: {objective}
- CTR Target: {ctr_target}%
- CPM Target: SGD {cpm_target}
- Budget: SGD {budget}
- Flight: {flight}

OVERALL PERFORMANCE:
- Impressions: {total_impressions:,}
- Clicks: {total_clicks:,}
- CTR: {ctr:.2f}%
- Spend: SGD {total_spend:,.2f}
- CPM: SGD {cpm:,.2f}
- CPC: SGD {cpc:,.2f}
- Conversions: {total_conversions:,}
- Conversion Rate: {conv_rate:.2f}%
- Cost per Conversion: SGD {cost_per_conv:,.2f}

--- BEGIN LINE ITEM DATA ---
{line_item_summary}

Please interpret the line item data above. Highlight top performers, weak spots, and strategic actions.
Only use the data above. Do not invent metrics. Write in natural, confident first person as if I ran the campaign myself.
"""

        response = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=0.85,
            messages=[
                {"role": "system", "content": "You write structured, strategic campaign insights like a confident media buyer."},
                {"role": "user", "content": prompt}
            ]
        )

        print("🔍 OpenAI Response:")
        print(response)

        report_text = response.choices[0].message.content if response.choices else "No response generated."
        return JSONResponse(content={"report": report_text})

    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})

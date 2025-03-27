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

# Load .env and setup FastAPI
load_dotenv()
app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI key setup
openai.api_key = os.getenv("OPENAI_API_KEY")
print("🔑 Loaded OpenAI API Key?", bool(openai.api_key))

# ---------- MAIN INSIGHT ENDPOINT ----------
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
        t0 = time.time()

        contents = await file.read()
        if len(contents) == 0:
            return JSONResponse(status_code=400, content={"error": "Uploaded file is empty."})

        df = pd.read_excel(BytesIO(contents))
        print("✅ Loaded DataFrame")
        print("📊 Columns:", df.columns.tolist())

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

        df[required_columns] = df[required_columns].fillna(0)
        df['Clicks'] = df['Clicks'].replace(0, 1)
        df['Spend'] = df['Spend'].replace(0, 0.01)

        total_impressions = df['Impressions'].sum()
        total_clicks = df['Clicks'].sum()
        total_spend = df['Spend'].sum()
        total_conversions = df['Total Conversions'].sum()

        ctr = round((total_clicks / total_impressions) * 100, 2) if total_impressions else 0
        cpm = round((total_spend / total_impressions) * 1000, 2) if total_impressions else 0
        cpc = round(total_spend / total_clicks, 2) if total_clicks else 0
        conv_rate = round((total_conversions / total_impressions) * 100, 2) if total_impressions else 0
        cost_per_conv = round(total_spend / total_conversions, 2) if total_conversions else 0
        print("✅ Calculated totals (CTR, CPM, etc.)")

        # LINE ITEM SUMMARY
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
            grouped = grouped.fillna(0).sort_values(by='Spend', ascending=False).head(5)

            summaries = []
            for _, row in grouped.iterrows():
                item = f"Line Item: {row['Line Item']} (IO: {row['Insertion Order']})\n"
                item += f"Impressions: {int(row['Impressions'])}, Clicks: {int(row['Clicks'])}, CTR: {row['CTR (%)']:.2f}%\n"
                item += f"Spend: SGD {row['Spend']:.2f}, CPM: SGD {row['CPM (SGD)']:.2f}, CPC: SGD {row['CPC (SGD)']:.2f}\n"
                item += f"Conversions: {int(row['Total Conversions'])}, Conversion Rate: {row['Conversion Rate (%)']:.2f}%"
                summaries.append(item)
            line_item_summary = "\n\n".join(summaries)
        print("✅ Line item summary complete")

        # HELPER for GROUP SUMMARIES
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

        creative_summary = build_group_summary(df, "Creative", "Creative")
        print("✅ Creative summary complete")
        device_summary = build_group_summary(df, "Device Type", "Device Type")
        print("✅ Device summary complete")
        os_summary = build_group_summary(df, "Device Model", "Device Model")
        print("✅ OS summary complete")

        # FINAL PROMPT
        prompt = f"""
You are a professional paid media strategist reporting on a DV360 campaign.

Your goal is to deliver a confident, data-driven, first-person performance report. You must sound like a human strategist who understands what the data means — not just repeat it. You are writing a commentary, not giving a speech.

### STRUCTURE OF YOUR REPORT

You must use the following format, including the headings exactly as shown:

## Executive Summary  
Summarize the overall results clearly. No need to mention targets here yet — just highlight standout results, surprises, and key wins/losses.

## Performance vs KPIs  
Discuss how we performed against CTR and CPM targets. Include insights into why those numbers might have occurred — was it audience, placement, creative, timing?

## Line Item Breakdown  
Analyze each top-spending line item (these are provided below).  
→ Instead of repeating long line item names, infer the **audience or intent** from the name. For example, "ViewQwest_Residential_Branding_Lifestyle_Android" might be summarized as "Lifestyle-oriented Android audiences".  
→ Compare segments (e.g. lifestyle vs. remarketing, Android vs. iOS)  
→ Highlight strategic takeaways. Always ask: **“So what?”** — Why did this perform or underperform? What does that tell us?

## Creative Performance  
Evaluate the top creatives based on CTR and engagement.  
→ Identify which ones worked and **why** (e.g. static vs animated, product vs lifestyle)  
→ Avoid just repeating creative filenames — try to interpret them as a strategist would

## Device & OS Analysis  
Discuss differences in performance between smartphones, tablets, desktops.  
→ Mention if one device type dominated  
→ If “Unknown” shows up, mention it tactfully and suggest we improve tracking or segmentation

## Conversion Analysis  
Was the campaign efficient in converting users?  
→ Mention conversion volume, rate, and cost per conversion  
→ Call out anything that stands out — high conversions from low CTR? Or vice versa?

## Strategic Observations & Recommendations  
End with forward-looking thoughts:  
→ What would you try next time?  
→ Which audiences or creatives to double down on?  
→ Any tracking, device targeting, or bid strategy changes you’d suggest?

---

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

## TOP LINE ITEM PERFORMANCE

{line_item_summary}

## TOP CREATIVES

{creative_summary}

## DEVICE PERFORMANCE

{device_summary}

## OS PERFORMANCE

{os_summary}
"""
 

        print("✅ Prompt built. Time so far:", round(time.time() - t0, 2), "seconds")
        print("🔍 Final Prompt Sent to GPT (truncated):\n", prompt[:500])

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


# ---------- CHAT INTERACT ENDPOINT ----------
class InteractionRequest(BaseModel):
    insight: str
    user_prompt: str
    mode: str

@app.post("/interact-insight/")
async def interact_with_insight(request: InteractionRequest):
    try:
        print("📥 Chat Request Received:", request.dict())
        print("🧠 Prompting OpenAI with:")
        print(f"Mode: {request.mode}\nInsight: {request.insight[:300]}...\nUser Prompt: {request.user_prompt}")

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

        result = response.choices[0].message.content if response.choices else "⚠️ No response from OpenAI."
        print("📤 Chat Result:", result[:300])
        return JSONResponse(content={"result": result})

    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})

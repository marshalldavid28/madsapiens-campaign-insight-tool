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
                content={"error": "Dataset is missing required columns.",
                         "missing_columns": missing_columns,
                         "available_columns": df.columns.tolist()}
            )

        # Clean numeric columns
        if df['Spend'].dtype == 'object':
            df['Spend'] = df['Spend'].str.replace('S$', '', regex=False).astype(float)

        df[required_columns] = df[required_columns].fillna(0)

        # Aggregate Totals
        total_impressions = df['Impressions'].sum()
        total_clicks = df['Clicks'].sum()
        total_spend = df['Spend'].sum()
        total_conversions = df['Total Conversions'].sum()

        ctr = round((total_clicks / total_impressions) * 100, 2) if total_impressions else 0
        cpm = round((total_spend / total_impressions) * 1000, 2) if total_impressions else 0
        cpc = round(total_spend / total_clicks, 2) if total_clicks else 0
        conv_rate = round((total_conversions / total_impressions) * 100, 2) if total_impressions else 0
        cost_per_conv = round(total_spend / total_conversions, 2) if total_conversions else 0

        print("✅ Totals calculated")

        def generate_group_summary(group_by, label):
            if group_by not in df.columns:
                return ""

            grouped = df.groupby(group_by).agg({
                'Impressions': 'sum',
                'Clicks': 'sum',
                'Spend': 'sum',
                'Total Conversions': 'sum'
            }).reset_index()

            grouped['CTR (%)'] = (grouped['Clicks'] / grouped['Impressions'].replace(0, pd.NA)) * 100
            grouped['CPM (SGD)'] = (grouped['Spend'] / grouped['Impressions'].replace(0, pd.NA)) * 1000
            grouped['CPC (SGD)'] = grouped.apply(lambda x: x['Spend'] / x['Clicks'] if x['Clicks'] else 0, axis=1)
            grouped['Conversion Rate (%)'] = (grouped['Total Conversions'] / grouped['Impressions'].replace(0, pd.NA)) * 100
            grouped = grouped.fillna(0).sort_values(by='Spend', ascending=False).head(5)

            summaries = []
            for _, row in grouped.iterrows():
                summaries.append(
                    f"{label}: {row[group_by]}\n"
                    f"Impressions: {int(row['Impressions'])}, Clicks: {int(row['Clicks'])}, CTR: {row['CTR (%)']:.2f}%\n"
                    f"Spend: SGD {row['Spend']:.2f}, CPM: SGD {row['CPM (SGD)']:.2f}, CPC: SGD {row['CPC (SGD)']:.2f}\n"
                    f"Conversions: {int(row['Total Conversions'])}, Conversion Rate: {row['Conversion Rate (%)']:.2f}%"
                )
            return "\n\n".join(summaries)

        line_item_summary = generate_group_summary("Line Item", "Line Item")
        creative_summary = generate_group_summary("Creative", "Creative")
        device_summary = generate_group_summary("Device Type", "Device Type")
        os_summary = generate_group_summary("Device Model", "Device Model")
        print("✅ All summaries generated")

        # FINAL PROMPT
        prompt = f"""
You are a professional paid media strategist reporting on a DV360 campaign.

Your goal is to deliver a confident, data-driven, first-person performance report. You must sound like a human strategist who understands what the data means — not just repeat it. You are writing a commentary, not giving a speech.

### STRUCTURE OF YOUR REPORT

Use the following sections and structure your insights accordingly:

## Executive Summary  
Summarize results. Don’t list metrics — highlight key wins, surprises, or inefficiencies.

## Performance vs KPIs  
Explain how the campaign did against CTR and CPM targets. Provide strategic hypotheses behind performance, not just facts.

## Line Item Breakdown  
Comment on audience targeting performance using the line item names provided below. Group and simplify — e.g. "Lifestyle Android users" or "Remarketing iOS". Focus on insights, not listing.

## Creative Performance  
Highlight top creatives and infer *why* they performed well — static vs video, product vs lifestyle.

## Device & OS Analysis  
Compare performance across smartphones, tablets, desktops. Mention if anything underperformed or needs attention (e.g., "Unknown" models).

## Conversion Analysis  
Discuss conversion volume, cost per conversion, and efficiency. Relate back to specific audiences or creatives if possible.

## Strategic Observations & Recommendations  
End with takeaways and what you would do differently next time. Be bold and helpful.

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
    mode: str  # "ask" or "edit"

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
            model="gpt-3.5-turbo",
            temperature=0.85,
            messages=messages
            print("🤖 Using GPT-3.5 for chat interaction")
        )

        result = response.choices[0].message.content if response.choices else "⚠️ No response from OpenAI."
        print("📤 Chat Result:", result[:300])
        return JSONResponse(content={"result": result})

    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})

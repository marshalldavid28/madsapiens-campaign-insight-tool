from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import pandas as pd
import openai
from io import BytesIO

app = FastAPI()

@app.post("/generate-insights/")
async def generate_insights(
    file: UploadFile = File(...),
    objective: str = Form(...),
    ctr_target: float = Form(...),
    cpm_target: float = Form(...),
    budget: float = Form(...),
    flight: str = Form(...),
    primary_metric: str = Form(...),
    secondary_metric: str = Form(...)
):
    try:
        contents = await file.read()
        df = pd.read_excel(BytesIO(contents))

        # Clean Spend column if necessary
        if 'Spend' in df.columns and df['Spend'].dtype == 'object':
            df['Spend'] = df['Spend'].str.replace('S$', '', regex=False).astype(float)

        # Calculate overall metrics
        total_impressions = df['Impressions'].sum()
        total_clicks = df['Clicks'].sum()
        total_spend = df['Spend'].sum()
        total_conversions = df['Total Conversions'].sum() if 'Total Conversions' in df.columns else 0

        ctr = round((total_clicks / total_impressions) * 100, 2) if total_impressions else 0
        cpm = round((total_spend / total_impressions) * 1000, 2) if total_impressions else 0
        cpc = round(total_spend / total_clicks, 2) if total_clicks else 0
        conv_rate = round((total_conversions / total_clicks) * 100, 2) if total_clicks else 0
        cost_per_conv = round(total_spend / total_conversions, 2) if total_conversions else 0

        # Group by Insertion Order and Line Item
        pivot_analysis = "### Performance Breakdown by Insertion Order & Line Item\n"

        if 'Insertion Order' in df.columns and 'Line Item' in df.columns:
            grouped = df.groupby(['Insertion Order', 'Line Item']).agg({
                'Impressions': 'sum',
                'Clicks': 'sum',
                'Spend': 'sum',
                'Total Conversions': 'sum' if 'Total Conversions' in df.columns else lambda x: 0
            }).reset_index()

            # Calculate additional metrics
            grouped['CTR (%)'] = (grouped['Clicks'] / grouped['Impressions']) * 100
            grouped['CPM (SGD)'] = (grouped['Spend'] / grouped['Impressions']) * 1000
            grouped['CPC (SGD)'] = grouped['Spend'] / grouped['Clicks']
            
            if 'Total Conversions' in df.columns:
                grouped['Conv Rate (%)'] = (grouped['Total Conversions'] / grouped['Clicks']) * 100
                grouped['Cost per Conv (SGD)'] = grouped['Spend'] / grouped['Total Conversions']

            grouped = grouped.fillna(0)

            # Formatting grouped data into structured breakdown
            for _, row in grouped.iterrows():
                pivot_analysis += f"\nInsertion Order: {row['Insertion Order']} | Line Item: {row['Line Item']}\n"
                pivot_analysis += f"- Impressions: {int(row['Impressions'])}\n"
                pivot_analysis += f"- Clicks: {int(row['Clicks'])}\n"
                pivot_analysis += f"- CTR: {row['CTR (%)']:.2f}%\n"
                pivot_analysis += f"- Spend: SGD {row['Spend']:.2f}\n"
                pivot_analysis += f"- CPM: SGD {row['CPM (SGD)']:.2f}\n"
                pivot_analysis += f"- CPC: SGD {row['CPC (SGD)']:.2f}\n"

                if 'Total Conversions' in df.columns:
                    pivot_analysis += f"- Conversions: {int(row['Total Conversions'])}\n"
                    pivot_analysis += f"- Conv Rate: {row['Conv Rate (%)']:.2f}%\n"
                    pivot_analysis += f"- Cost per Conv: SGD {row['Cost per Conv (SGD)']:.2f}\n"

        # Prepare the AI prompt
        prompt = f"""
You are a paid digital advertising strategist analyzing a DV360 campaign.

Write a professional, confident report summarizing campaign performance, focusing on the **primary metric**: {primary_metric}. Also provide additional insights on the **secondary metric**: {secondary_metric}. Ensure the tone is structured, corporate, and data-driven.

## Executive Summary:
Summarize key performance outcomes. 

## Overall Planned vs Delivered:
Compare planned vs actual performance based on the budget, CTR target, and CPM target.

## Line Item Observations:
Analyze performance at a granular level by insertion order and line item. Identify high- and low-performing segments.

## Conversion Analysis:
Provide insights into conversion trends, cost efficiency, and optimization suggestions.

## Recommendations / Strategy Updates:
Suggest logical next steps based on the data.

### **Campaign Data:**
- Objective: {objective}
- Budget: SGD {budget}
- Flight: {flight}
- **Primary Focus Metric:** {primary_metric}
- **Secondary Metric:** {secondary_metric}
- CTR Target: {ctr_target}%
- CPM Target: SGD {cpm_target}

### **Overall Performance:**
- Impressions: {total_impressions}
- Clicks: {total_clicks}
- CTR: {ctr}%
- Spend: SGD {total_spend}
- CPM: SGD {cpm}
- CPC: SGD {cpc}
- Conversions (Landing Page Visits): {total_conversions}
- Conversion Rate: {conv_rate}%
- Cost per Conversion: SGD {cost_per_conv}

{pivot_analysis}

Write in a clear, concise, and professional manner.
"""

        # Call OpenAI API to generate insights
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

    except Exception as e:  # ✅ Now correctly indented inside `try`
        import traceback
        error_details = traceback.format_exc()  # Get full error details
        print("ERROR:", error_details)  # Print the error in logs
        return JSONResponse(status_code=500, content={"error": str(e), "details": error_details})

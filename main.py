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

        # Debugging: Log received inputs
        print("🔹 Received Inputs:")
        print(f"Primary Metric: {primary_metric}")
        print(f"Secondary Metric: {secondary_metric}")

        # Log available columns in the uploaded file
        available_columns = df.columns.tolist()
        print("🔹 Available Columns in Uploaded File:", available_columns)

        # Clean Spend column if necessary
        if 'Spend' in df.columns and df['Spend'].dtype == 'object':
            df['Spend'] = df['Spend'].str.replace('S$', '', regex=False).astype(float)

        # Standardized column mapping
        metric_mapping = {
            "CTR": "CTR (%)",
            "CPM": "CPM (SGD)",
            "CPC": "CPC (SGD)",
            "Conversions": "Total Conversions",
            "Conv Rate": "Conv Rate (%)",
            "Cost per Conv": "Cost per Conv (SGD)"
        }

        # Get the correct column names
        primary_metric_col = metric_mapping.get(primary_metric, primary_metric)
        secondary_metric_col = metric_mapping.get(secondary_metric, secondary_metric)

        # Validate metric columns
        missing_columns = []
        if primary_metric_col not in available_columns:
            missing_columns.append(primary_metric)
        if secondary_metric_col not in available_columns:
            missing_columns.append(secondary_metric)

        if missing_columns:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "One or more selected metrics were not found in the dataset.",
                    "missing_metrics": missing_columns,
                    "available_columns": available_columns
                }
            )

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
        if 'Insertion Order' in df.columns and 'Line Item' in df.columns:
            grouped = df.groupby(['Insertion Order', 'Line Item']).agg({
                'Impressions': 'sum',
                'Clicks': 'sum',
                'Spend': 'sum',
                primary_metric_col: 'sum',
                secondary_metric_col: 'sum'
            }).reset_index()

            # Sort by Primary Metric to find Top 5 & Bottom 5 performers
            sorted_grouped = grouped.sort_values(by=primary_metric_col, ascending=False)

            top_performers = sorted_grouped.head(5)
            bottom_performers = sorted_grouped.tail(5)

            pivot_analysis = "### Top 5 Performing Line Items:\n"
            for _, row in top_performers.iterrows():
                pivot_analysis += f"- {row['Insertion Order']} | {row['Line Item']}: {row[primary_metric_col]:.2f}\n"

            pivot_analysis += "\n### Bottom 5 Performing Line Items:\n"
            for _, row in bottom_performers.iterrows():
                pivot_analysis += f"- {row['Insertion Order']} | {row['Line Item']}: {row[primary_metric_col]:.2f}\n"
        else:
            pivot_analysis = "No Insertion Order or Line Item data found."

        # Prepare the AI prompt (limit text length to avoid token overuse)
        prompt = f"""
You are a paid digital advertising strategist analyzing a DV360 campaign.

Write a structured, professional report, focusing on the **primary metric**: {primary_metric} and the **secondary metric**: {secondary_metric}.

### **Key Performance Metrics**
- CTR Target: {ctr_target}%
- CPM Target: SGD {cpm_target}
- Budget: SGD {budget}
- Flight Period: {flight}

### **Overall Performance Summary**
- Impressions: {total_impressions:,}
- Clicks: {total_clicks:,}
- CTR: {ctr:.2f}%
- Spend: SGD {total_spend:,.2f}
- CPM: SGD {cpm:,.2f}
- CPC: SGD {cpc:,.2f}
- Conversions: {total_conversions}
- Conversion Rate: {conv_rate:.2f}%
- Cost per Conversion: SGD {cost_per_conv:,.2f}

{pivot_analysis[:2000]}  # Limit breakdown details to avoid exceeding token limit

Please write a concise, data-driven report, keeping each section brief and to the point.
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

    except Exception as e:  
        import traceback
        error_details = traceback.format_exc()  
        print("ERROR:", error_details)  
        return JSONResponse(status_code=500, content={"error": str(e), "details": error_details})

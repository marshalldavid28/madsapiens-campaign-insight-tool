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

        # Clean the spend column if needed
        if 'Spend' in df.columns and df['Spend'].dtype == 'object':
            df['Spend'] = df['Spend'].str.replace('S$', '', regex=False).astype(float)

        # Calculate overall metrics (as before)
        total_impressions = df['Impressions'].sum()
        total_clicks = df['Clicks'].sum()
        total_spend = df['Spend'].sum()
        total_conversions = df['Total Conversions'].sum() if 'Total Conversions' in df.columns else 0

        ctr = round((total_clicks / total_impressions) * 100, 2) if total_impressions else 0
        cpm = round((total_spend / total_impressions) * 1000, 2) if total_impressions else 0
        cpc = round(total_spend / total_clicks, 2) if total_clicks else 0
        conv_rate = round((total_conversions / total_clicks) * 100, 2) if total_clicks else 0
        cost_per_conv = round(total_spend / total_conversions, 2) if total_conversions else 0

        # Create pivot-style analysis by Insertion Order and Line Item
        pivot_analysis = ""
        
        # Check if the necessary columns exist for grouping
        groupby_columns = []
        if 'Insertion Order' in df.columns:
            groupby_columns.append('Insertion Order')
        if 'Line Item' in df.columns:
            groupby_columns.append('Line Item')
            
        if groupby_columns:
            # Group by Insertion Order and/or Line Item and calculate metrics
            grouped = df.groupby(groupby_columns).agg({
                'Impressions': 'sum',
                'Clicks': 'sum',
                'Spend': 'sum',
                'Total Conversions': 'sum' if 'Total Conversions' in df.columns else lambda x: 0
            }).reset_index()
            
            # Calculate derived metrics for each group
            grouped['CTR (%)'] = round((grouped['Clicks'] / grouped['Impressions']) * 100, 2)
            grouped['CPM (SGD)'] = round((grouped['Spend'] / grouped['Impressions']) * 1000, 2)
            grouped['CPC (SGD)'] = round(grouped['Spend'] / grouped['Clicks'], 2)
            
            if 'Total Conversions' in df.columns:
                grouped['Conv Rate (%)'] = round((grouped['Total Conversions'] / grouped['Clicks']) * 100, 2)
                grouped['Cost per Conv (SGD)'] = round(grouped['Spend'] / grouped['Total Conversions'], 2)
            
            # Format the grouped data for the prompt
            pivot_analysis = "Detailed Breakdown:\n"
            
            # Handle potential NaN values
            grouped = grouped.fillna(0)
            
            for _, row in grouped.iterrows():
                group_name = " > ".join([f"{col}: {row[col]}" for col in groupby_columns])
                pivot_analysis += f"\n{group_name}\n"
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

        prompt = f"""
You are a digital strategist who personally ran a DV360 campaign.

Write a professional, confident, first-person report summarizing campaign performance. Ensure that you dont come across as too excited. 
Phrases like "Im proud to announce" should be avoided. Speak corporate, to the point and professional. Should not sound conversational.

Good example:

"The campaign has performed well this month with a CTR of x%, which is y times higher than the planned CTR which was z%."

Bad example:

"I am happy to note that the campaign has done really well, despite some challenge" - this is too conversational and storytelling-esque.

Be detailed and structured. Use a clear sectioned format like this:

1. Executive Summary
2. Overall Planned vs Delivered
3. Line Item Observations
4. Conversion Analysis
5. Recommendations / Strategy Updates


Some tips for each of the above sections:

1. Executive Summary: Make sure that, while this is a summary, it has some of the key data points that the client would want to know. A good way to think about this is, if the client is doesnt have time to read the rest of the insights, this section should highlight the most important outcomes.
2. Overall Planned vs Delivered: Make sure that his section compares what was expected to what was delivered. Keep it sharp and to the point, but feel free to corporate in making this part sound good if the perforamnce was above expectations.
3. Line Item Observations:  This is usually where the audience targeting layer sits. So analyse the line item performance and ensure to speak about this in sgranular detail. Which audience performed better, what was the performance, what do we recommend to the client, any reason why this audience may have performend better for this brand etc are questions you should consider.
4. Conversion Analysis - Go into data detail with this. Explain what the performance was and leave some space for the user to add their own thoughts.

Use the data below. Where helpful, feel free to explain performance trends logically, based only on available metrics. Do not guess optimizations — but you may add two or three possible contributing factors (clearly marked).

Data:
- Objective: {objective}
- Budget: SGD {budget}
- Flight: {flight}
- CTR Target: {ctr_target}%
- CPM Target: SGD {cpm_target}

Overall Performance:
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

Write in first person, like a strategist explaining this to a team or client. Be sharp, confident, and clear. Keep it realistic — not too fluffy. Expand on the insights where useful.

In the Line Item Observations section, analyze the performance of each Insertion Order and Line Item, comparing their metrics against each other and against the overall campaign KPIs. Identify top and bottom performers, unusual patterns, and potential optimization opportunities.
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

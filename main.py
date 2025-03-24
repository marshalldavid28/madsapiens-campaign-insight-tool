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
        
        # Print column names for debugging (can be removed in production)
        print(f"Columns in uploaded Excel: {df.columns.tolist()}")
        
        # Flexible column name handling - check for variations
        impressions_col = next((col for col in df.columns if 'impression' in col.lower()), 'Impressions')
        clicks_col = next((col for col in df.columns if 'click' in col.lower() and 'ctr' not in col.lower()), 'Clicks')
        spend_col = next((col for col in df.columns if 'spend' in col.lower() or 'cost' in col.lower()), 'Spend')
        conversions_col = next((col for col in df.columns 
                              if any(x in col.lower() for x in ['conversion', 'conv', 'action'])), 
                              'Total Conversions')
        
        # Clean the spend column if needed
        if df[spend_col].dtype == 'object':
            df[spend_col] = df[spend_col].str.replace('S$', '', regex=False).astype(float)
        
        # Calculate overall metrics with flexible column names
        total_impressions = df[impressions_col].sum()
        total_clicks = df[clicks_col].sum()
        total_spend = df[spend_col].sum()
        total_conversions = df[conversions_col].sum() if conversions_col in df.columns else 0

        # Safe calculations to handle division by zero
        ctr = round((total_clicks / total_impressions) * 100, 2) if total_impressions else 0
        cpm = round((total_spend / total_impressions) * 1000, 2) if total_impressions else 0
        cpc = round(total_spend / total_clicks, 2) if total_clicks else 0
        conv_rate = round((total_conversions / total_clicks) * 100, 2) if total_clicks else 0
        cost_per_conv = round(total_spend / total_conversions, 2) if total_conversions else 0

        # Create pivot-style analysis by Insertion Order and Line Item
        pivot_analysis = ""
        
        # Check for common dimension column names with flexible matching
        dimension_columns = {
            'Insertion Order': next((col for col in df.columns 
                                  if any(x in col.lower() for x in ['insertion order', 'io', 'campaign'])), 
                                  None),
            'Line Item': next((col for col in df.columns 
                             if any(x in col.lower() for x in ['line item', 'li', 'ad group'])), 
                             None)
        }
        
        # Filter out None values
        groupby_columns = [col_name for col_name, col in dimension_columns.items() if col is not None]
        actual_columns = [dimension_columns[col] for col in groupby_columns]
            
        if actual_columns:
            # Create a mapping dictionary for flexible column names
            col_mapping = {
                dimension_columns[col]: col for col in groupby_columns
            }
            col_mapping[impressions_col] = 'Impressions'
            col_mapping[clicks_col] = 'Clicks'
            col_mapping[spend_col] = 'Spend'
            if conversions_col in df.columns:
                col_mapping[conversions_col] = 'Total Conversions'
            
            # Rename columns for analysis
            analysis_df = df.rename(columns=col_mapping)
            
            # Group by Insertion Order and/or Line Item and calculate metrics
            agg_cols = {
                'Impressions': 'sum',
                'Clicks': 'sum',
                'Spend': 'sum'
            }
            
            if 'Total Conversions' in analysis_df.columns:
                agg_cols['Total Conversions'] = 'sum'
                
            grouped = analysis_df.groupby(groupby_columns).agg(agg_cols).reset_index()
            
            # Calculate derived metrics for each group
            grouped['CTR (%)'] = grouped.apply(
                lambda x: round((x['Clicks'] / x['Impressions']) * 100, 2) if x['Impressions'] > 0 else 0, 
                axis=1
            )
            grouped['CPM (SGD)'] = grouped.apply(
                lambda x: round((x['Spend'] / x['Impressions']) * 1000, 2) if x['Impressions'] > 0 else 0, 
                axis=1
            )
            grouped['CPC (SGD)'] = grouped.apply(
                lambda x: round(x['Spend'] / x['Clicks'], 2) if x['Clicks'] > 0 else 0, 
                axis=1
            )
            
            if 'Total Conversions' in grouped.columns:
                grouped['Conv Rate (%)'] = grouped.apply(
                    lambda x: round((x['Total Conversions'] / x['Clicks']) * 100, 2) if x['Clicks'] > 0 else 0, 
                    axis=1
                )
                grouped['Cost per Conv (SGD)'] = grouped.apply(
                    lambda x: round(x['Spend'] / x['Total Conversions'], 2) if x['Total Conversions'] > 0 else 0, 
                    axis=1
                )
            
            # Add performance indicators compared to targets
            grouped['CTR vs Target'] = grouped['CTR (%)'] - ctr_target
            grouped['CPM vs Target'] = cpm_target - grouped['CPM (SGD)']  # Inverted since lower CPM is better
            
            # Sort by performance (CTR)
            grouped = grouped.sort_values('CTR (%)', ascending=False)
            
            # Format the grouped data for the prompt
            pivot_analysis = "Detailed Breakdown (sorted by CTR performance):\n"
            
            # Handle potential NaN values
            grouped = grouped.fillna(0)
            
            for _, row in grouped.iterrows():
                group_name = " > ".join([f"{col}: {row[col]}" for col in groupby_columns])
                pivot_analysis += f"\n{group_name}\n"
                pivot_analysis += f"- Impressions: {int(row['Impressions'])}\n"
                pivot_analysis += f"- Clicks: {int(row['Clicks'])}\n"
                pivot_analysis += f"- CTR: {row['CTR (%)']:.2f}% ({'+' if row['CTR vs Target'] >= 0 else ''}{row['CTR vs Target']:.2f}% vs target)\n"
                pivot_analysis += f"- Spend: SGD {row['Spend']:.2f}\n"
                pivot_analysis += f"- CPM: SGD {row['CPM (SGD)']:.2f} ({'+' if row['CPM vs Target'] >= 0 else ''}{row['CPM vs Target']:.2f} vs target)\n"
                pivot_analysis += f"- CPC: SGD {row['CPC (SGD)']:.2f}\n"
                
                if 'Total Conversions' in grouped.columns:
                    pivot_analysis += f"- Conversions: {int(row['Total Conversions'])}\n"
                    pivot_analysis += f"- Conv Rate: {row['Conv Rate (%)']:.2f}%\n"
                    pivot_analysis += f"- Cost per Conv: SGD {row['Cost per Conv (SGD)']:.2f}\n"

        prompt = f"""
You are a paid digital advertising campaign specialist who personally ran a DV360 campaign.

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
        print(f"Error in generate_insights: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

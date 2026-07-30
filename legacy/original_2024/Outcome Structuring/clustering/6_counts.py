import pandas as pd

def value_count_column(file_path, column_name, output_file=None):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Perform value counts on the specified column
    value_counts = df[column_name].value_counts()

    # Convert to a DataFrame for better handling
    value_counts_df = value_counts.reset_index()
    value_counts_df.columns = [column_name, 'Count']  # Rename columns

    # Save to a new file if output_file is provided
    if output_file:
        value_counts_df.to_csv(output_file, index=False)
        print(f"Value counts saved to {output_file}")
    
    # Return the value counts DataFrame
    return value_counts_df



categories = """0 Mortality - This outcome category focuses on death rates or time until death as the primary measure. Clinical trials often assess whether an intervention reduces overall mortality or the risk of death from a specific cause.
    1 Infection - This category includes the incidence, rate, or severity of infections. Studies might measure how well an intervention prevents infections, reduces their frequency, or shortens their duration.
    2 Pain - Pain-related outcomes assess changes in pain intensity, frequency, or severity. Trials evaluate if a treatment alleviates pain, improves pain management, or reduces patients' reliance on pain-related medications.
    3 Mental health - Outcomes in this category cover emotional well-being, including changes in depression, anxiety, mood disorders, or psychological distress. Researchers examine whether an intervention improves mental health scores or reduces symptoms of mental illness.
    4 Function - Function encompasses patients' ability to perform daily activities or maintain mobility. This can include improvements in walking, self-care, work capability, or performance in activities of daily living.
    5 Quality of Life - Quality of Life measures focus on overall well-being, satisfaction, and comfort. Trials may use standardized questionnaires to assess changes in patients' physical, emotional, and social functioning, as well as their perceived life satisfaction.
    6 Psychosocial - Psychosocial outcomes look at social interaction, family dynamics, community engagement, self-esteem, and coping skills. These outcomes help determine if a treatment influences a patient's social roles, relationships, and personal identity.
    7 Compliance with treatment - This category measures how consistently and accurately patients follow prescribed therapies, medication regimens, or lifestyle recommendations. It can also examine factors that improve adherence and barriers that reduce compliance.
    8 Adverse events - Adverse event outcomes track the occurrence, severity, and types of unwanted side effects or complications related to an intervention. By assessing adverse events, researchers gauge the safety profile of a treatment.
    9 Satisfaction with care - This outcome is about patients' subjective assessment of their treatment experience. It includes satisfaction with healthcare providers, communication, convenience, and the overall healthcare process.
    10 Resource use - These outcomes focus on healthcare utilization, such as hospital stays, outpatient visits, medication consumption, and costs. Trials may measure whether an intervention is more or less resource-intensive compared to standard care.
    11 Device/intervention failure - This category examines the reliability and durability of a medical device or the effectiveness of a specific procedure. It tracks how often a device malfunctions, requires replacement, or if an intervention does not produce the intended effect.
    12 Physiological or clinical - Physiological or clinical outcomes measure direct biological or clinical parameters. These may include changes in blood pressure, heart rate, lab values (like cholesterol or blood glucose), or clinical markers of disease progression, providing objective evidence of treatment impact.
"""

# Example usage
file_path = 'results_cls-0-2053.csv'
column_name = 'class'  # Replace with your column name
output_file = 'value_counts.csv'

result_df = value_count_column(file_path, column_name, output_file)

# Print the result
print(result_df)
print(result_df['Count'].sum())

labels = [cat.split('-')[0].strip() for cat in categories.split('\n') if cat != '']
print(labels)

for index, row in result_df.iterrows():
    if row['class'].isnumeric() and int(row['class']) < 13:
        print(labels[int(row['class'])], row['Count'])
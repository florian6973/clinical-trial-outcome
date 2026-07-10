import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'trials.csv'

# Read the data and handle date parsing
data = pd.read_csv(file_path)

# Preprocess and clean the data
unique_trials = data[['nct_id', 'start_month_year', 'snomed_top_level_term', 'object']].drop_duplicates()

# Convert dates to YYYY-MM format
def standardize_date(date_str):
    try:
        return pd.to_datetime(date_str).strftime('%Y-%m')
    except:
        return None

# Clean the dates
unique_trials['start_month_year'] = unique_trials['start_month_year'].apply(standardize_date)
unique_trials['start_month_year'] = pd.to_datetime(unique_trials['start_month_year'], format='%Y-%m')

# Get total counts across all years for each SNOMED term (distinct trials)
snomed_counts = unique_trials.groupby('snomed_top_level_term')['nct_id'].nunique().sort_values(ascending=False)
print("Top 10 SNOMED terms by total count:")
print(snomed_counts.head(10))

# Plot 1: Horizontal bar chart
plt.figure(figsize=(6, 10))  # Half as wide
ax = snomed_counts.head(20).plot(kind='barh', width=0.8)
plt.title('Top 20 SNOMED Terms by Number of Unique Trials', pad=20)
plt.xlabel('Number of Unique Trials')
plt.ylabel('SNOMED Top Level Term')
plt.yticks(range(20), snomed_counts.head(20).index)  # Remove [::-1] to keep correct order
plt.grid(True, linestyle='--', alpha=0.3)
plt.savefig('trials_by_snomed.png', dpi=300, bbox_inches='tight')
plt.close()

# Filter data for 2005-2022
mask = (unique_trials['start_month_year'].dt.year >= 2005) & \
       (unique_trials['start_month_year'].dt.year <= 2021)
filtered_trials = unique_trials[mask]

# Get top terms excluding Miscellanea
snomed_counts_filtered = snomed_counts[snomed_counts.index != 'Miscellanea']
top_terms = snomed_counts_filtered.head(4).index  # Get top 4 excluding Miscellanea
top_terms_data = filtered_trials[filtered_trials['snomed_top_level_term'].isin(top_terms)]

# Group by year and SNOMED term (fix the grouping here)
trend_data = top_terms_data.groupby([
    top_terms_data['start_month_year'].dt.year,
    'snomed_top_level_term'
])['nct_id'].nunique().unstack(fill_value=0)

# Plot 2: Line chart with yearly data
plt.figure(figsize=(15, 8))
trend_data.plot(linewidth=2, marker='o', markersize=6)
plt.title('Trial Trends by Year for Top 5 SNOMED Terms (2005-2022)', pad=20)
plt.xlabel('Year')
plt.ylabel('Number of Unique Trials')
plt.legend(title='SNOMED Top Level Term', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)

# Improve x-axis readability with clean year labels
plt.xticks(trend_data.index, 
          [int(year) for year in trend_data.index],
          rotation=45,
          ha='right')

plt.subplots_adjust(right=0.85, bottom=0.15)
plt.savefig('trial_trends_snomed.png', dpi=300, bbox_inches='tight')
plt.close()

# Create separate trend charts for each top SNOMED category
plt.figure(figsize=(20, 15))
fig, axes = plt.subplots(2, 2, figsize=(20, 15))  # 2x2 grid for 4 SNOMED categories
fig.suptitle('Outcome Trends by Top SNOMED Categories (2005-2022)', fontsize=16, y=0.95)

# Flatten axes for easier iteration
axes_flat = axes.flatten()

# For each top SNOMED category
for idx, snomed_term in enumerate(top_terms[:4]):
    print(f"Processing {snomed_term}")
    
    # Filter data for this SNOMED category
    category_data = filtered_trials[filtered_trials['snomed_top_level_term'] == snomed_term]
    
    # Clean and combine outcomes
    category_data['object'] = category_data['object'].replace({
       'objective response rate': 'response rate',
        'overall response rate': 'response rate',
        'objective response': 'response rate',
        'response': 'response rate',
        'overall survival os': 'overall survival',
        'progression free survival pfs': 'progression free survival',
        'survival': 'overall survival',
        'response': 'response rate',
        'duration of response': 'response rate',
        'best overall response': 'response rate',
        'overall response': 'response rate',
        'adverse event': 'adverse events',
        'plasma': 'plasma concentration',
        'maximum plasma concentration': 'plasma concentration',
        'observed plasma': 'plasma concentration',
        'total score': 'score'
    })
    
    # Remove unwanted outcomes
    category_data = category_data[~category_data['object'].isin([
          'curve', 
        'incidence',
        'area under the concentration time curve',
        'safety',
        'duration',
        'serious',
        'positive'    
    ])]
    
    # Get top 5 outcomes for this specific SNOMED category (using unique trials)
    category_outcomes = category_data.groupby('object')['nct_id'].nunique().sort_values(ascending=False).head(5).index
    
    # Get total trials per year for this category
    yearly_totals = category_data.groupby(category_data['start_month_year'].dt.year)['nct_id'].nunique()
    
    # Group by year and outcome term (counting unique trials)
    category_trends = category_data[category_data['object'].isin(category_outcomes)].groupby([
        category_data['start_month_year'].dt.year,
        'object'
    ])['nct_id'].nunique().unstack(fill_value=0)
    
    # Convert to percentages
    category_trends_pct = category_trends.div(yearly_totals, axis=0) * 100
    
    # Create individual plot
    plt.figure(figsize=(12, 8))
    category_trends_pct.plot(linewidth=2, marker='o', markersize=6)
    
    # Customize plot
    plt.title(f'Outcome Trends in {snomed_term} Trials (2005-2022)', pad=20)
    plt.xlabel('Year')
    plt.ylabel('Percentage of Trials')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Outcome Term', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Format x-axis
    plt.xticks(category_trends_pct.index, 
              [int(year) for year in category_trends_pct.index],
              rotation=45,
              ha='right')
    
    # Adjust layout
    plt.subplots_adjust(right=0.85, bottom=0.15)
    
    # Save individual plot
    plt.savefig(f'outcome_trends_{snomed_term.lower().replace("/", "_")}.png', 
                dpi=300, 
                bbox_inches='tight')
    plt.close()

# Create analysis for outcome changes
def calculate_period_percentages(data, snomed_term, outcome):
    # Filter for the specific SNOMED term
    snomed_data = data[data['snomed_top_level_term'] == snomed_term]
    
    # Calculate early period (before 2015)
    early_mask = snomed_data['start_month_year'].dt.year < 2015
    early_total = snomed_data[early_mask]['nct_id'].nunique()
    early_outcome = snomed_data[
        early_mask & (snomed_data['object'] == outcome)
    ]['nct_id'].nunique()
    
    # Calculate late period (2015+)
    late_mask = snomed_data['start_month_year'].dt.year >= 2015
    late_total = snomed_data[late_mask]['nct_id'].nunique()
    late_outcome = snomed_data[
        late_mask & (snomed_data['object'] == outcome)
    ]['nct_id'].nunique()
    
    # Calculate percentages
    early_pct = (early_outcome / early_total * 100) if early_total > 0 else 0
    late_pct = (late_outcome / late_total * 100) if late_total > 0 else 0
    
    return early_pct, late_pct, late_pct - early_pct

# Get top 100 SNOMED-outcome pairs first
pair_counts = filtered_trials.groupby(['snomed_top_level_term', 'object'])['nct_id'].nunique()
top_100_pairs = pair_counts.sort_values(ascending=False).head(100)

# Calculate changes for each pair
changes = []
for (snomed_term, outcome) in top_100_pairs.index:
    early_pct, late_pct, change = calculate_period_percentages(filtered_trials, snomed_term, outcome)
    changes.append({
        'SNOMED Term': snomed_term,
        'Outcome': outcome,
        'Early %': early_pct,
        'Late %': late_pct,
        'Change': change,
        'Total Trials': pair_counts[(snomed_term, outcome)]
    })

# Create DataFrame and sort by absolute change to get top 20 largest shifts
changes_df = pd.DataFrame(changes)
changes_df = changes_df.sort_values('Change', key=abs, ascending=False).head(20)  # Take top 20 by magnitude
changes_df = changes_df.sort_values('Change', key=abs, ascending=True)  # Resort for plotting

# Create horizontal bar chart
plt.figure(figsize=(12, 10))
bars = plt.barh(range(len(changes_df)), changes_df['Change'])
plt.yticks(range(len(changes_df)), 
          [f"{row['SNOMED Term']}\n{row['Outcome']}" for _, row in changes_df.iterrows()],
          fontsize=8)

# Color bars based on direction of change
for i, bar in enumerate(bars):
    if changes_df.iloc[i]['Change'] < 0:
        bar.set_color('#d62728')  # red for decrease
    else:
        bar.set_color('#2ca02c')  # green for increase

plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.grid(True, linestyle='--', alpha=0.3)
plt.title('Largest Changes in Outcome Usage (2015+ vs Before 2015)\nFrom Top 100 SNOMED-Outcome Pairs', pad=20)
plt.xlabel('Change in Percentage Points')

# Add annotations
for i, row in enumerate(changes_df.iterrows()):
    row = row[1]
    plt.text(row['Change'], i, 
            f"{row['Change']:.1f}%\n({row['Early %']:.1f}% → {row['Late %']:.1f}%)", 
            va='center',
            ha='left' if row['Change'] >= 0 else 'right',
            fontsize=8)

plt.tight_layout()
plt.savefig('outcome_changes.png', dpi=300, bbox_inches='tight')
plt.close()

# For each of our top 4 SNOMED categories
for snomed_term in top_terms[:4]:
    print(f"Processing changes for {snomed_term}")
    
    # Filter data for this SNOMED category
    category_data = filtered_trials[filtered_trials['snomed_top_level_term'] == snomed_term]
    
    # Clean and combine outcomes
    category_data['object'] = category_data['object'].replace({
         'objective response rate': 'response rate',
        'overall response rate': 'response rate',
        'objective response': 'response rate',
        'response': 'response rate',
        'overall survival os': 'overall survival',
        'progression free survival pfs': 'progression free survival',
        'survival': 'overall survival',
        'response': 'response rate',
        'duration of response': 'response rate',
        'best overall response': 'response rate',
        'overall response': 'response rate',
        'adverse event': 'adverse events',
        'plasma': 'plasma concentration',
        'maximum plasma concentration': 'plasma concentration',
        'observed plasma': 'plasma concentration',
        'total score': 'score'
    })
    
    # Remove unwanted outcomes
    category_data = category_data[~category_data['object'].isin([
          'curve', 
        'incidence',
        'area under the concentration time curve',
        'safety',
        'duration',
        'serious',
        'positive'    
    ])]
    
    # Get top 5 outcomes for this specific SNOMED category (same as trends)
    category_outcomes = category_data.groupby('object')['nct_id'].nunique().sort_values(ascending=False).head(5).index
    
    # Pre-calculate period masks once
    early_mask = category_data['start_month_year'].dt.year < 2015
    late_mask = category_data['start_month_year'].dt.year >= 2015
    
    # Get total trials in each period
    early_total = category_data[early_mask]['nct_id'].nunique()
    late_total = category_data[late_mask]['nct_id'].nunique()
    
    # Calculate changes for the top 5 outcomes only
    changes = []
    for outcome in category_outcomes:
        # Count trials with this outcome in each period
        early_outcome = category_data[early_mask & (category_data['object'] == outcome)]['nct_id'].nunique()
        late_outcome = category_data[late_mask & (category_data['object'] == outcome)]['nct_id'].nunique()
        
        # Calculate percentages
        early_pct = (early_outcome / early_total * 100) if early_total > 0 else 0
        late_pct = (late_outcome / late_total * 100) if late_total > 0 else 0
        change = late_pct - early_pct
        
        changes.append({
            'Outcome': outcome,
            'Early %': early_pct,
            'Late %': late_pct,
            'Change': change,
            'Total Trials': category_data[category_data['object'] == outcome]['nct_id'].nunique()
        })
    
    # Create DataFrame and sort by magnitude of change
    changes_df = pd.DataFrame(changes)
    changes_df = changes_df.sort_values('Change', key=abs, ascending=True)
    
    # Create horizontal bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.barh(range(len(changes_df)), changes_df['Change'])
    plt.yticks(range(len(changes_df)), 
              [f"{row['Outcome']}" for _, row in changes_df.iterrows()],
              fontsize=10)
    
    # Color bars based on direction of change
    for i, bar in enumerate(bars):
        if changes_df.iloc[i]['Change'] < 0:
            bar.set_color('#d62728')  # red for decrease
        else:
            bar.set_color('#2ca02c')  # green for increase
    
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.title(f'Largest Changes in Outcome Usage for {snomed_term} (2015+ vs Before 2015)', pad=20)
    plt.xlabel('Change in Percentage Points')
    
    # Add annotations
    for i, row in enumerate(changes_df.iterrows()):
        row = row[1]
        plt.text(row['Change'], i, 
                f"{row['Change']:.1f}%\n({row['Early %']:.1f}% → {row['Late %']:.1f}%)", 
                va='center',
                ha='left' if row['Change'] >= 0 else 'right',
                fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'outcome_changes_{snomed_term.lower().replace("/", "_")}.png', 
                dpi=300, 
                bbox_inches='tight')
    plt.close()


# Read the studies file and merge with our existing data
studies_df = pd.read_csv('studies.txt', delimiter='|')
filtered_trials_with_phase = filtered_trials.merge(
    studies_df[['nct_id', 'phase']], 
    on='nct_id', 
    how='left'
)





#######OUTCOME TRENDS BY PHASE AND SNOMED

# Get top 5 SNOMED terms (excluding Miscellanea) and specify phases
top_snomed = snomed_counts_filtered.head(5).index
selected_phases = ['PHASE1', 'PHASE2', 'PHASE3']  # Explicitly specify the phases we want

# Create a 2x2 grid of plots (2 SNOMED conditions x 2 phases)
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('Outcome Trends by Phase and SNOMED Category (2005-2022)', fontsize=16, y=0.95)

# Iterate through SNOMED conditions and phases
for i, snomed_term in enumerate(top_snomed):
    for j, phase in enumerate(selected_phases):
        print(f"Processing {snomed_term} - {phase}")
        
        # Filter data for this SNOMED category and phase
        phase_data = filtered_trials_with_phase[
            (filtered_trials_with_phase['snomed_top_level_term'] == snomed_term) &
            (filtered_trials_with_phase['phase'] == phase)
        ]
        
        # Clean and combine outcomes
        phase_data['object'] = phase_data['object'].replace({
        'objective response rate': 'response rate',
        'overall response rate': 'response rate',
        'objective response': 'response rate',
        'response': 'response rate',
        'overall survival os': 'overall survival',
        'progression free survival pfs': 'progression free survival',
        'survival': 'overall survival',
        'response': 'response rate',
        'duration of response': 'response rate',
        'best overall response': 'response rate',
        'overall response': 'response rate',
        'adverse event': 'adverse events',
        'plasma': 'plasma concentration',
        'maximum plasma concentration': 'plasma concentration',
        'observed plasma': 'plasma concentration',
        'total score': 'score',
        })
        
        # Remove unwanted outcomes
        phase_data = phase_data[~phase_data['object'].isin([
            'curve', 
        'incidence',
        'area under the concentration time curve',
        'safety',
        'duration',
        'serious',
        'positive',
        'severity'
        ])]
        
        # Get top 5 outcomes specifically for this combination
        phase_outcomes = phase_data.groupby('object')['nct_id'].nunique().sort_values(ascending=False).head(5).index
        
        # Get total trials per year
        yearly_totals = phase_data.groupby(phase_data['start_month_year'].dt.year)['nct_id'].nunique()
        
        # Group by year and outcome term
        phase_trends = phase_data[phase_data['object'].isin(phase_outcomes)].groupby([
            phase_data['start_month_year'].dt.year,
            'object'
        ])['nct_id'].nunique().unstack(fill_value=0)
        
        # Convert to percentages
        phase_trends_pct = phase_trends.div(yearly_totals, axis=0) * 100
        
        # Apply 3-year rolling average
        smoothed_trends = phase_trends_pct.rolling(window=3, center=True, min_periods=1).mean()
        
        # Create new figure for each combination
        plt.figure(figsize=(15, 8))
        
        # Plot both raw data (light) and smoothed trends (bold)
        for column in phase_trends_pct.columns:
            # Plot raw data as light dots
            plt.plot(phase_trends_pct.index, phase_trends_pct[column], 
                    'o', markersize=4, alpha=0.3, label=f'{column} (raw)')
            # Plot smoothed trend as bold line
            plt.plot(smoothed_trends.index, smoothed_trends[column], 
                    linewidth=3, label=f'{column} (3-year avg)')
        
        # Customize plot
        plt.title(f'Outcome Trends in {snomed_term} - {phase} Trials (2005-2022)', pad=20)
        plt.xlabel('Year')
        plt.ylabel('Percentage of Trials')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Format x-axis
        plt.xticks(phase_trends_pct.index,
                  [int(year) for year in phase_trends_pct.index],
                  rotation=45,
                  ha='right')
        
        # Add legend below the plot
        handles, labels = plt.gca().get_legend_handles_labels()
        # Keep only the smoothed trend labels (every other label)
        handles = handles[1::2]
        labels = [label.replace(' (3-year avg)', '') for label in labels[1::2]]
        plt.legend(handles, labels,
                  title='Outcome Term', 
                  loc='upper center',
                  bbox_to_anchor=(0.5, -0.15),
                  ncol=3)
        
        # Adjust layout with more bottom margin for legend
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        
        # Save plot
        plt.savefig(f'outcome_trends_{snomed_term.lower().replace("/", "_")}_{phase.lower()}.png', 
                    dpi=300, 
                    bbox_inches='tight')
        plt.close()







# Get top 2 SNOMED terms (excluding Miscellanea)
top_snomed = snomed_counts_filtered.head(2).index
selected_phases = ['PHASE1', 'PHASE2', 'PHASE3']

# Increase font sizes globally
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'legend.title_fontsize': 16
})

for snomed_term in top_snomed:
    print(f"Processing {snomed_term}")
    
    # Filter data for this SNOMED category and phases
    snomed_data = filtered_trials_with_phase[
        (filtered_trials_with_phase['snomed_top_level_term'] == snomed_term) &
        (filtered_trials_with_phase['phase'].isin(selected_phases))
    ].copy()  # Create a copy to avoid warnings
    
    # Clean and combine outcomes
    outcome_mapping = {
        'objective response rate': 'response rate',
        'overall response rate': 'response rate',
        'objective response': 'response rate',
        'response': 'response rate',
        'overall survival os': 'overall survival',
        'progression free survival pfs': 'progression free survival',
        'survival': 'overall survival',
        'response': 'response rate',
        'duration of response': 'response rate',
        'best overall response': 'response rate',
        'overall response': 'response rate',
        'adverse event': 'adverse events',
        'plasma': 'plasma concentration',
        'maximum plasma concentration': 'plasma concentration',
        'observed plasma': 'plasma concentration',
        'total score': 'score'
    }
    
    # Create a copy before modifying
    snomed_data = snomed_data.copy()
    snomed_data['object'] = snomed_data['object'].replace(outcome_mapping)
    
    # Remove unwanted outcomes
    unwanted_outcomes = [
        'curve', 
        'incidence',
        'area under the concentration time curve',
        'safety',
        'duration',
        'serious',
        'positive',
        'severity'
    ]
    
    snomed_data = snomed_data[~snomed_data['object'].isin(unwanted_outcomes)]
    
    # Get top 10 outcomes for this SNOMED category across all phases
    # Sort in descending order (most common first)
    top_outcomes = snomed_data.groupby('object')['nct_id'].nunique().sort_values(ascending=True).tail(10).index
    
    # Create figure (taller for more outcomes)
    plt.figure(figsize=(15, 12))  # Increased figure size
    
    # Calculate percentages for all phases
    phase_pcts = []
    for phase in selected_phases:
        phase_data = snomed_data[snomed_data['phase'] == phase]
        total_trials = len(phase_data['nct_id'].unique())
        
        outcome_pcts = []
        for outcome in top_outcomes:
            outcome_trials = len(phase_data[phase_data['object'] == outcome]['nct_id'].unique())
            pct = (outcome_trials / total_trials) * 100
            outcome_pcts.append(pct)
        phase_pcts.append(outcome_pcts)
    
    # Plot horizontal bars
    y = range(len(top_outcomes))
    height = 0.25
    
    # Plot bars in reverse order (Phase 3 at bottom, Phase 1 at top)
    plt.barh([i - height for i in y], phase_pcts[2], height, 
             label='PHASE3', alpha=0.8)
    plt.barh([i for i in y], phase_pcts[1], height, 
             label='PHASE2', alpha=0.8)
    plt.barh([i + height for i in y], phase_pcts[0], height, 
             label='PHASE1', alpha=0.8)
    
    # Customize plot with larger fonts
    plt.title(f'Top 10 Outcomes in {snomed_term} Trials by Phase', pad=20, fontsize=20)
    plt.xlabel('Percentage of Trials', fontsize=16)
    plt.ylabel('Outcome', fontsize=16)
    plt.yticks(y, top_outcomes, fontsize=14)
    
    # Add legend with larger font
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = handles[::-1]
    labels = labels[::-1]
    plt.legend(handles, labels, title='Trial Phase', fontsize=14, title_fontsize=16)
    
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add percentage labels on the bars with larger font
    for i in range(len(top_outcomes)):
        # Phase 3 label (bottom)
        plt.text(phase_pcts[2][i], i - height, 
                f'{phase_pcts[2][i]:.1f}%', 
                va='center', ha='left',
                fontsize=12)
        # Phase 2 label (middle)
        plt.text(phase_pcts[1][i], i, 
                f'{phase_pcts[1][i]:.1f}%', 
                va='center', ha='left',
                fontsize=12)
        # Phase 1 label (top)
        plt.text(phase_pcts[0][i], i + height, 
                f'{phase_pcts[0][i]:.1f}%', 
                va='center', ha='left',
                fontsize=12)
    
    # Adjust layout with more padding
    plt.tight_layout(pad=1.5)
    
    # Save plot
    plt.savefig(f'outcome_comparison_{snomed_term.lower().replace("/", "_")}.png', 
                dpi=300, 
                bbox_inches='tight')
    plt.close()
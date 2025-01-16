import pandas as pd
from .fHALF import fHALF
import os

# Get the current file's directory (Utils directory)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the CSV file
csv_path = os.path.join(current_dir, '12koutput.csv')

# Read the CSV file
y = pd.read_csv(csv_path)

print("Columns in the DataFrame:", y.columns)

# Convert `organ` and `symptom` columns to lowercase
y['organ'] = y['organ'].str.lower()
y['symptom'] = y['symptom'].str.lower()

# Sort the dataset by `organ` and `symptom`
z = y.sort_values(by=['organ', 'symptom']).reset_index(drop=True)

# Group by `symptom` and `organ` to calculate the count
final_ds = (
    z.groupby(['symptom', 'organ'])
    .size()
    .reset_index(name='cnt')  # Add a `cnt` column for the count
)

# Sort by `cnt` in descending order
f = final_ds.sort_values(by='cnt', ascending=False).reset_index(drop=True)

# Remove rows with `unspecified` organ or invalid symptom values
fs = f[
    (f['organ'] != 'unspecified') &
    (f['symptom'] != 'organ') &
    (f['symptom'] != 'symptom')
].reset_index(drop=True)

# Display the final DataFrame
print(fs)

def rec_filter(input_string, filtered_data):
    # Use fHALF to parse the input string into a DataFrame
    symptoms_list = fHALF(input_string)
    symptoms_df = pd.DataFrame(symptoms_list, columns=['symptom', 'duration', 'organ'])
    symptoms_df['duration'] = symptoms_df['duration'].astype(int)  # Convert duration to integer

    # Perform INNER JOIN on fs and symptoms_df based on organ and symptom
    our_vals = pd.merge(fs, symptoms_df, on=['organ', 'symptom'], how='inner')

    # Group by organ to calculate the sum of cnt
    organ_group = our_vals.groupby('organ').agg(summation=('cnt', 'sum')).reset_index()

    # Calculate the denominator for p of organs (sum of all summations)
    denominator_for_p_of_organs = organ_group['summation'].sum()

    # Merge our_vals with organ_group to calculate naive_bayes_probability_factor
    our_vals = pd.merge(our_vals, organ_group, on='organ', how='left')
    our_vals['naive_bayes_probability_factor'] = our_vals['cnt'] / our_vals['summation']

    # Sort our_vals by naive_bayes_probability_factor in descending order
    our_vals_sorted = our_vals.sort_values(by='naive_bayes_probability_factor', ascending=False)

    # Group by symptom and select the row with the maximum naive_bayes_probability_factor
    final_group = our_vals_sorted.loc[
        our_vals_sorted.groupby('symptom')['naive_bayes_probability_factor'].idxmax()
    ].reset_index(drop=True)

    # Select relevant columns for the final DataFrame
    rec_filter_df = final_group[['symptom', 'duration', 'organ']]

    # Create a DataFrame for "nil organ" symptoms
    temp_organs = pd.DataFrame({'symptom': ['nil organ'], 'organ': ['nil organ']})

    # Perform FULL OUTER JOIN between rec_filter_df and temp_organs
    final_result = pd.merge(
        rec_filter_df, temp_organs, on='symptom', how='outer', suffixes=('', '_temp')
    ).fillna({'duration': 0, 'organ': 'nil organ'})

    # Convert the DataFrame to a list of tuples and return
    return list(final_result[['symptom', 'duration', 'organ']].itertuples(index=False, name=None))

# # Example Usage
# fs_data = {
#     'symptom': ['symp1', 'symp2', 'symp3', 'symp1'],
#     'organ': ['org1', 'org2', 'org3', 'org1'],
#     'cnt': [10, 20, 30, 15]
# }
# fs = pd.DataFrame(fs_data)
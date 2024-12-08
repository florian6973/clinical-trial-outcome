# read json outputs/chatgpt_zero_4_raw.json

# for each element of the list, split by comma, and add 
# list to a new dict with key Quantity or Object of Interest

# outcomes are in samples_gpt.txt (one per line)
# save json with same name and _preprocessed suffix
# with open(...', 'w') as f:
#     json.dump(list(zip(outcomes, outputs_parsed, [None]*len(outcomes), [None]*len(outcomes))), f)


import json

# Read the JSON file
input_file = 'outputs/chatgpt_zero_4_raw.json'
output_file = 'outputs/chatgpt_zero_4_raw_preprocessed.json'
samples_file = 'samples_gpt.txt'

# Load JSON data
with open(input_file, 'r') as f:
    data = json.load(f)

# Parse each element in the list
parsed_data = []
for element in data['Quantity or Object of Interest']:
    parsed_element = {}
    parts = element.split(', ')
    parsed_element['Quantity or Object of Interest'] = parts
    parsed_data.append(parsed_element)

# Read outcomes from the samples file
with open(samples_file, 'r') as f:
    outcomes = [line.strip() for line in f.readlines()]

# Ensure the outcomes and parsed data are aligned
if len(outcomes) != len(parsed_data):
    raise ValueError("The number of outcomes does not match the parsed data length.")

# Create the final JSON structure
final_data = list(zip(outcomes, parsed_data, [None] * len(outcomes), [None] * len(outcomes)))

# Save the preprocessed data to a new JSON file
with open(output_file, 'w') as f:
    json.dump(final_data, f, indent=4)

print(f"Preprocessed data saved to {output_file}")
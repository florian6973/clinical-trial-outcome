

import json
from bs4 import BeautifulSoup
import re

def parse_html_to_json(html_input):
    # Parse the HTML
    soup = BeautifulSoup(html_input, "html.parser")

    # Extract elements within the <span> tags with the specified class

    # Construct JSON structure
    result = \
        {
            "Time".lower(): [],
            "Quantity Unit".lower(): [],
            "Quantity Measure".lower(): [],
            "Quantity or object of Interest".lower(): [],
            "Additional constraints".lower(): [],
            "Quantity range".lower(): []
        }
    
    for key in result:
        items_of_interest = [span.text.strip() for span in soup.find_all("span", class_=
                                                                         re.compile(key, re.I))]
                                                                         #key)]
        result[key] = items_of_interest
        

    
    # Convert to JSON string for output
    # return json.dumps(result, indent=2)
    return result

with open('outputs_html.json', 'r') as f:
    data = json.load(f)

output_json = []
for data_pred in data:
    try:
        # Execute function and print the result
        json_output = parse_html_to_json(data_pred)
        print(data_pred)
        print(json_output)
        output_json.append(json_output)
    except:
        output_json.append({"error"})

with open('outputs_json.json', 'w') as f:
    json.dump(output_json, f)


# https://github.com/BIDS-Xu-Lab/Clinical_Entity_Recognition_Using_GPT_models
import json

import pandas as pd


data_path = "./arxiv-metadata-oai-snapshot.json"

categories_to_check = ["astro-ph","physics", "gr-qc", "hep", "cond-mat"]

# Initialize a list to store data
astro_ph_data = []

# Load the JSON file line by line
with open(data_path, "r") as file:
    for line in file:
        # Parse each line as JSON
        try:
            entry_data = json.loads(line)
        except json.JSONDecodeError:
            # Skip lines that cannot be parsed as JSON
            continue
        
        # Check if the paper belongs to the "astro-ph" category
        if "categories" in entry_data and (any(cat in entry_data["categories"] for cat in categories_to_check)):
            # Extract the relevant information
            paper_info = {
                "id": entry_data.get("id", ""),
                "title": entry_data.get("title", ""),
                "authors": entry_data.get("authors"),
                "abstract": entry_data.get("abstract", ""),
                "category": entry_data.get("categories", ""),
                "link": f"https://arxiv.org/abs/{entry_data['id']}"
                # Add more fields as needed
            }
            # Append the paper information to the list
            astro_ph_data.append(paper_info)

# Create a DataFrame from the collected data
df = pd.DataFrame(astro_ph_data)
print(df.head()["abstract"])
print(len(df))

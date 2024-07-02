from typing import Optional
import json

import pandas as pd


data_path = "../data/arxiv-metadata-oai-snapshot.json"
# data_path = "../data/arxiv-metadata-oai-cs-10k.json"

# categories_to_check = ["astro-ph","physics", "gr-qc", "hep", "cond-mat"]
# categories_to_check = ["cs"]

def filter_data_by_category(categories: list[str], limit: Optional[int] = None):

    # Initialize a list to store data
    filtered_data = []

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
            if "categories" in entry_data and (any(cat in entry_data["categories"] for cat in categories)):
                filtered_data.append(entry_data)
                # Extract the relevant information
                # paper_info = {
                #     "id": entry_data.get("id", ""),
                #     "title": entry_data.get("title", ""),
                #     "authors": entry_data.get("authors"),
                #     "abstract": entry_data.get("abstract", ""),
                #     "category": entry_data.get("categories", ""),
                #     "link": f"https://arxiv.org/abs/{entry_data['id']}"
                #     # Add more fields as needed
                # }
                # # Append the paper information to the list
                # filtered_data.append(paper_info)
            
            if limit and len(filtered_data) >= limit:
                break

    return filtered_data

    # Create a DataFrame from the collected data
    df = pd.DataFrame(filtered_data)
    return df


def save_data(data: list, filename: str):
    with open(filename, "w") as file:
        for row in data:
            file.write(json.dumps(row) + "\n")

if __name__ == "__main__":
    data = filter_data_by_category(["cs"])
    print(len(data))
    print(data[0])
    # save_data(data, "../data/arxiv-metadata-oai-resnet.json")

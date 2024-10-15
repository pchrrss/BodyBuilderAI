import json
import os

# Function to convert JSON exercise data to Alpaca format
def convert_to_alpaca(json_data):
    # Extract data from JSON
    name = json_data.get("name", "Unknown Exercise")
    instructions = json_data.get("instructions", [])
    primary_muscles = ', '.join(json_data.get("primaryMuscles", []))
    secondary_muscles = ', '.join(json_data.get("secondaryMuscles", []))
    category = json_data.get("category", "General")
    level = json_data.get("level", "all levels")

    # Construct Alpaca style prompt and response
    prompt = f"Create a workout plan for {name}, a {level} level exercise in the {category} category. The primary muscles worked are {primary_muscles} and the secondary muscles are {secondary_muscles}."
    response = "\n".join(instructions)

    # Create Alpaca entry
    alpaca_entry = {
        "instruction": prompt,
        "input": "",
        "output": response
    }

    return alpaca_entry

# Directory containing JSON exercise files
input_directory = "json"

# Open a file to write JSONL data
with open("alpaca/output_alpaca.jsonl", "w") as outfile:
    # Iterate through all JSON files in the specified directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".json"):
            file_path = os.path.join(input_directory, filename)
            with open(file_path, "r") as file:
                json_data = json.load(file)
                alpaca_entry = convert_to_alpaca(json_data)
                # Write each entry as a JSON line
                outfile.write(json.dumps(alpaca_entry) + "\n")

print("Conversion complete! The Alpaca formatted data has been saved to 'output_alpaca.json'.")

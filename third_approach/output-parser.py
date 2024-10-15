from pydantic import BaseModel, Field, ValidationError

# Define the Pydantic schema
class OutputSchema(BaseModel):
    field1: str = Field(..., description="first field")
    field2: str = Field(..., description="second field")

# Define a function to manually parse and validate the model output
def parse_output(output):
    try:
        parsed_output = OutputSchema(**output)
        return parsed_output
    except ValidationError as e:
        print("Validation Error:", e)
        return None

# Example output from the model (as a dictionary)
model_output = {
    "field1": "value1",
    "field2": "value2"
}

# Parse and validate the output
parsed_output = parse_output(model_output)

if parsed_output:
    print(parsed_output)
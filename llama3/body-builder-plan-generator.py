import requests
import json
import random

ageRanges = ['18-29', '30-39', '40-49', '50-59', '60+']
bodyTypes = ['slim', 'average', 'heavy']
goals = ['gain muscle mass', 'get shredded', 'lose weight', 'maintain fitness']
fatRanges = ['5-9%', '10-14%', '15-19%', '20-24%', '25-29%', '30-34%', '35-39%', '40+%']
focusAreas = ['legs', 'belly', 'arms', 'chest', 'back', 'full body']
fitnessLevels = list(range(1, 11))
equipments = ['no equipment', 'basic equipment', 'full equipment']
timesPerWeeks = list(range(1, 7))

workoutType = ["strength training", "HIIT and strength training", ]

ageRange = random.choice(ageRanges)
goal = random.choice(goals)
fatRange = random.choice(fatRanges)
focusArea = random.choice(focusAreas)
fitnessLevel = random.choice(fitnessLevels)
equipment = random.choice(equipments)
timesPerWeek = random.choice(timesPerWeeks)

model = "llama3:latest"

prompt = (f"""generate one realistically believable fitness plan for a person with the following characteristics:
          {{
            workoutType: workoutType,
            intensity: intensity,
            frequencyIntensity: frequencyIntensity,
            equipment: equipmentExercises,
            focusExercises: focusExercises,
            recommendedPlan: 'workoutType' with 'equipmentExercises', focusing on 'focusArea' with 'intensity' intensity, 'timesPerWeek' times per week.`,
          }}
          Respond using JSON. Key names should have no backslashes, values should use plain ascii with no special characters.
          """)

data = {
    "prompt": prompt,
    "model": model,
    "format": "json",
    "stream": False,
    "options": {"temperature": 2.5, "top_p": 0.99, "top_k": 100},
}

print(f"""
    Generate the recommended fitness plan for this input:
    {{
        ageRange: {ageRange},
        goal: {goal},
        fatRange: {fatRange},
        focusArea: {focusArea},
        fitnessLevel: {fitnessLevel},
        equipment: {equipment},
        timesPerWeek: {timesPerWeek}
    }}
    """)
try:
    response = requests.post("http://localhost:11434/api/generate", json=data, stream=False)

    # Check if the response was successful
    if response.status_code == 200:
        # Parse the JSON response
        json_data = response.json()

        # Check if the 'response' key is in the JSON
        if "response" in json_data:
            # Print the generated JSON data nicely formatted
            print(json.dumps(json.loads(json_data["response"]), indent=2))
        else:
            # If 'response' key is not found, print the entire JSON response
            print("Key 'response' not found in the response:")
            print(json.dumps(json_data, indent=2))
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(f"Response content: {response.text}")
except requests.exceptions.RequestException as e:
    # Handle connection errors or timeouts
    print(f"An error occurred: {e}")
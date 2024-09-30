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

ageRange = random.choice(ageRanges)
bodyType = random.choice(bodyTypes)
goal = random.choice(goals)
goal2 = random.choice(goals)
fatRange = random.choice(fatRanges)
focusArea = random.choice(focusAreas)
fitnessLevel = random.choice(fitnessLevels)
equipment = random.choice(equipments)
timesPerWeek = random.choice(timesPerWeeks)

model = "body-builder-model:latest"

prompt = (f"""
            I am a {ageRange} years old person with a {bodyType} body type. My goal is to {goal}, and I want to focus on my {goal} and {goal2}. Currently, my body fat percentage is around {fatRange}, and my fitness level is {fitnessLevel}/10. I have access to {equipment}, and I plan to work out {timesPerWeek} days a week.
            Can you generate a fitness plan for me in JSON format based on these details, including the following structure:
            * workout_plan: an array of for each day per week I plan to work out, with:
                * day: a number from 1 to 7 representing the day of the week (from monday to sunday).
                * focus_area: the main focus area for the training.
                * exercises: an array of exercises for the day, With:
                    * name: the name of the exercise.
                    * instruction: give some instructions to how apply correctly the exercise.
                    * sets: the number of sets for the exercise.
                    * reps: the number of repetitions per set.
          """)

# Verify the JSON is valid before sending it back, in.

data = {
    "prompt": prompt,
    "model": model,
    "format": "json",
    "stream": False,
    "options": {"temperature": 0.1, "top_p": 0.99, "top_k": 100},
}

print(f"prompt = {prompt}")

try:
    response = requests.post("http://localhost:11434/api/generate", json=data, stream=False)

    # Check if the response was successful
    if response.status_code == 200:
        # Parse the JSON response
        json_data = response.json()

        # Check if the 'response' key is in the JSON
        if "response" in json_data:
            # print(f"response = {json_data["response"]}")
            # Print the generated JSON data nicely formatted
            print(f"output = {json.dumps(json.loads(json_data["response"]), indent=2)}")
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
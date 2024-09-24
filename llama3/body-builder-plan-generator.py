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
bodyType = random.choice(bodyTypes)
goal = random.choice(goals)
goal2 = random.choice(goals)
fatRange = random.choice(fatRanges)
focusArea = random.choice(focusAreas)
fitnessLevel = random.choice(fitnessLevels)
equipment = random.choice(equipments)
timesPerWeek = random.choice(timesPerWeeks)

model = "llama3:latest"

prompt = (f"""
            I am a {ageRange} years old person with a {bodyType} body type. My goal is to {goal}, and I want to focus on my {goal} and {goal2}. Currently, my body fat percentage is around {fatRange}, and my fitness level is {fitnessLevel}/10. I have access to {equipment}, and I plan to work out {timesPerWeek} days a week.
            Can you generate a fitness plan for me in JSON format based on these details, including the following structure:
            * age_range: my age range.
            * body_type: my body type.
            * goal: my fitness goal.
            * focus_areas: areas I want to focus on.
            * body_fat_range: my body fat range.
            * fitness_level: my current fitness level.
            * equipment: the equipment I have access to.
            * workout_frequency_per_week: the number of days I will work out each week.
            * workout_plan: a breakdown of the workouts for each day per week I plan to work out, with:
                * day: the day of the week.
                * focus_area: the main focus area for the day.
                * exercises: a list of exercises for the day, With:
                    * name: the name of the exercise.
                    * sets: the number of sets for the exercise.
                    * reps: the number of repetitions per set.
            The output should be structured as a JSON object with detailed values for each field. Key names should have no backslashes, values should use plain ascii with no special characters.
          """)

data = {
    "prompt": prompt,
    "model": model,
    "format": "json",
    "stream": False,
    "options": {"temperature": 2.5, "top_p": 0.99, "top_k": 100},
}

print(f"{prompt}")

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
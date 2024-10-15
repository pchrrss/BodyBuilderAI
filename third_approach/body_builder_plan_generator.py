import random

from call_llama import call_llama

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

model = "body-builder-alpaca-model:latest"
restdays = 7 - timesPerWeek

prompt = (f"""
            I am a {ageRange} years old person with a {bodyType} body type. My goal is to {goal}, and I want to focus on my {goal} and {goal2}. Currently, my body fat percentage is around {fatRange}, and my fitness level is {fitnessLevel}/10. I have access to {equipment}, and I plan to work out {timesPerWeek} days a week.
            Can you generate a fitness plan for me in JSON format based on these details, including the following structure:
            * workout_plan: an array of 7 elements, with {timesPerWeek} days of training and {restdays} rest days, with:
                * day: a string representing week's day of the training.
                * focus_area: the main focus area for the training, if it's a rest day, the focus area will 'Rest day'.
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

call_llama(data)
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

model = "body-builder-llama3-1:latest"
restdays = 7 - timesPerWeek

prompt = (f"""
            I am a {ageRange} years old person with a {bodyType} body type. My goal is to {goal}, and I want to focus on my {goal} and {goal2}. Currently, my body fat percentage is around {fatRange}, and my fitness level is {fitnessLevel}/10. I have access to {equipment}, and I plan to work out {timesPerWeek} days a week.
            Can you give a random tips in one line, to apply during my training, with:
            * tips: the advice to apply during the training
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
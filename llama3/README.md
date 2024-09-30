# Use Llama3
## Install Ollama on mac
https://ollama.com/download
## run llama3
```sh
ollama run llama3
```

# Create your own model
## Create your input file
```sh
# set the base model
FROM llama3:8b
# set the custom parameter values
PARAMETER temperature 1
PARAMETER num_ctx 4096

# set the system message
SYSTEM When you receive a message with the following input
```
## Create your new model
```sh
ollama create my-model -f my-file
```
To create the body builder model: 
```sh
ollama create body-builder-model -f bodyBuilderLlam3Model
```

# Create a python program to generate a fitness plan
## Example of output
```sh
% python body-builder-plan-generator.py

prompt:
- workout_plan: an array of for each day per week I plan to work out, with:
	-  day: a number from 1 to 7 representing the day of the week (from monday to sunday)
		- focus_area: the main focus area for the training.
		- exercises: an array of exercises for the day, With:
			- name: the name of the exercise.
			- instruction: give some instructions to how apply correctly the exercise.
			- sets: the number of sets for the exercise.
			- reps: the number of repetitions per set.

Example of output
"workout_plan": [
    {
      "day": 2,
      "focus_area": "Cardio",
      "exercises": [
        {
          "name": "Brisk Walking",
          "instruction": "Walk at a brisk pace for 30 minutes, aiming for a moderate intensity.",
          "sets": 1,
          "reps": null
        }
      ]
    },
    {
      "day": 3,
      "focus_area": "Strength Training",
      "exercises": [
        {
          "name": "Squats",
          "instruction": "Stand with feet shoulder-width apart, then bend knees and lower body until thighs are parallel to the ground. Push back up to starting position.",
          "sets": 3,
          "reps": 12
        },
        {
          "name": "Push-ups",
          "instruction": "Start in a plank position with hands shoulder-width apart, then lower body until chest nearly touches the ground. Push back up to starting position.",
          "sets": 3,
          "reps": 10
        }
      ]
    }
]
```
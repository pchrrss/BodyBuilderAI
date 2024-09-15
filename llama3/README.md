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
# Create a python program to generate a fitness plan
## Example of output
```sh
% python body-builder-plan-generator.py

    Generate the recommended fitness plan for this input:
    {
        ageRange: 30-39,
        goal: lose weight,
        fatRange: 40+%,
        focusArea: full body,
        fitnessLevel: 8,
        equipment: full equipment,
        timesPerWeek: 5
    }

{
  "workoutType": "Cardio and Strength",
  "intensity": "Moderate to High",
  "frequencyIntensity": {
    "Low to Medium Impact": 3,
    "High Intensity": 1
  },
  "equipmentExercises": [
    {
      "treadmillRun": true,
      "rowMachineRow": true,
      "ellipticaltrainerWorkout": true,
      "freeweightliftingWeights": false
    }
  ],
  "focusExercises": [
    "Increasing Distance Running"
  ]
}
```
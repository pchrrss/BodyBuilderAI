from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 2x faster
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # 4bit for 405b!
    "unsloth/Mistral-Small-Instruct-2409",     # Mistral 22b 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!

    "unsloth/Llama-3.2-1B-bnb-4bit",           # NEW! Llama 3.2 models
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct", # or choose "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# data preparation
from unsloth.chat_templates import get_chat_template
from datasets import Dataset
from transformers import AutoTokenizer

# Charger le tokenizer pour le modèle (par exemple, LLaMA 3.2)
tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct")

# Adapter le tokenizer avec le template de chat
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1"  # Assure-toi d'utiliser le bon template en fonction de ton modèle.
)

# Fonction pour lire le fichier et créer un dataset
def create_dataset_from_text(file_path):
    # Lire le fichier texte contenant les conversations
    with open(file_path, 'r') as f:
        content = f.read()

    # Diviser le contenu en blocs de conversations (chaque conversation étant séparée par une double nouvelle ligne)
    conversations = content.strip().split("\n\n")

    # Préparer les exemples pour le dataset
    # Adapter le format pour correspondre à une structure de messages avec rôles
    examples = []
    for convo in conversations:
        # Diviser chaque bloc de conversation par les démarcations <|im_start|> et <|im_end|>
        # pour identifier les rôles et le contenu.
        messages = []
        mytext = convo;
        for block in convo.split("<|im_start|>")[1:]:
            parts = block.split("\n", 1)
            if len(parts) == 2:
                role = parts[0].strip()  # Récupérer le rôle (par exemple 'user' ou 'assistant')
                content = parts[1].replace("<|im_end|>", "").strip()  # Récupérer le contenu sans la balise de fin
                messages.append({"role": role, "content": content})

        # Ajouter la conversation formatée à la liste des exemples
        examples.append({"messages": messages, "text": mytext})

    # Créer un objet Dataset à partir des exemples
    dataset = Dataset.from_list(examples)

    # Retourner le dataset
    return dataset

# Créer le dataset à partir du fichier "fitness_conversation.txt"
dataset = create_dataset_from_text('data/fitness_conversation.txt')

# Fonction de formatage pour préparer chaque exemple pour le modèle
def formatting_prompts_func(examples):
    # Appliquer le template de chat pour chaque conversation sans tokeniser
    texts = []
    for messages in examples["messages"]:
        # Utiliser le tokenizer pour appliquer le template à chaque message
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}

def formatting_prompts_func2(examples):
    texts = []
    for messages in examples["messages"]:
        # Si "messages" est une liste, concaténer les messages en une seule chaîne de caractères
        if isinstance(messages, list):
            # Concaténer les messages avec des séparateurs pour les distinguer
            text = "\n".join([f"<|im_start|>{msg['role']}\n{msg['content']}\n<|im_end|>" for msg in messages])
        else:
            text = messages  # Si "messages" est déjà une chaîne, l'utiliser directement.

        # Appliquer le template de chat au texte concaténé
        formatted_text = tokenizer.apply_chat_template(text, tokenize=False, add_generation_prompt=False)
        texts.append(formatted_text)

    return {"text": texts}

# Adapter la fonction de formatage pour créer la structure de messages attendue
def formatting_prompts_func3(examples):
    texts = []
    for messages in examples["messages"]:
        # Si "messages" est une liste, on construit les messages attendus
        if isinstance(messages, list):
            # Créer une liste de dictionnaires avec les rôles et les contenus
            structured_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]

            # Appliquer le template de chat au format structuré
            formatted_text = tokenizer.apply_chat_template(structured_messages, tokenize=False, add_generation_prompt=False)
        else:
            raise ValueError("Expected 'messages' to be a list of dictionaries with 'role' and 'content' keys.")

        texts.append(formatted_text)

    return {"text": texts}

def formatting_prompts_func4(examples):
    # List to store formatted outputs
    formatted_outputs = []

    # Parcourir chaque exemple (qui est une liste de messages)
    for messages in examples["messages"]:
        # Construire la liste de messages avec rôles et contenus
        formatted_messages = [
            {"role": message["role"], "content": message["content"]}
            for message in messages
        ]

        # Construire la version textuelle pour l'entraînement
        # Ajout de balises et formatage spécifique
        text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        text += "Cutting Knowledge Date: December 2023\n"
        text += "Today Date: 26 July 2024\n\n"
        text += "<|eot_id|>"

        for message in formatted_messages:
            text += f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n"
            text += f"{message['content']}<|eot_id|>"

        # Ajouter la structure formatée au résultat
        formatted_output = {
            "messages": formatted_messages,
            "text": text
        }
        formatted_outputs.append(formatted_output)

    return formatted_outputs

def formatting_prompts_func5(examples):
    # Listes pour stocker les résultats
    formatted_messages = []
    formatted_texts = []

    # Parcourir chaque exemple (qui est une liste de messages)
    for messages in examples["messages"]:
        # Construire la liste de messages avec rôles et contenus
        structured_messages = [
            {"role": message["role"], "content": message["content"]}
            for message in messages
        ]

        # Construire la version textuelle pour l'entraînement
        text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        text += "Cutting Knowledge Date: December 2023\n"
        text += "Today Date: 26 July 2024\n\n"
        text += "<|eot_id|>"

        for message in structured_messages:
            text += f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n"
            text += f"{message['content']}<|eot_id|>"

        # Ajouter la structure formatée aux résultats
        formatted_messages.append(structured_messages)
        formatted_texts.append(text)

    # Retourner un dictionnaire avec les champs 'messages' et 'text'
    return {
        "messages": formatted_messages,
        "text": formatted_texts
    }

def formatting_prompts_func6(examples):
    # Listes pour stocker les conversations
    formatted_conversations = []

    # Parcourir chaque exemple (qui est une liste de messages)
    for messages in examples["messages"]:
        # Construire la liste de messages avec rôles et contenus
        structured_messages = [
            {"role": message["role"], "content": message["content"]}
            for message in messages
        ]

        # Construire la version textuelle pour l'entraînement
        text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        text += "Cutting Knowledge Date: December 2023\n"
        text += "Today Date: 26 July 2024\n\n"
        text += "<|eot_id|>"

        for message in structured_messages:
            text += f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n"
            text += f"{message['content']}<|eot_id|>"

        # Créer la structure d'une conversation
        formatted_conversation = {
            "messages": structured_messages,
            "text": text
        }

        # Ajouter la conversation à la liste
        formatted_conversations.append(formatted_conversation)

    # Retourner un dictionnaire avec la clé 'conversations'
    return {
        "conversations": formatted_conversations
    }


# Appliquer la fonction de formatage sur le dataset
formatted_dataset = dataset.map(formatting_prompts_func6, batched=True)

# Afficher un échantillon pour vérifier
print(formatted_dataset[0])



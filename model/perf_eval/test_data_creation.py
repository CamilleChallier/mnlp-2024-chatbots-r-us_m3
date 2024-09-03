import json

# ===========================================
# Load the different test datasets 
# ===========================================

# Intel’s Orca DPO Pairs Dataset
#? https://huggingface.co/datasets/Intel/orca_dpo_pairs
orca_test_data = []
with open("model/datasets/dpo_orca_test.jsonl", "r") as file:
    for line in file:
        orca_test_data.append(json.loads(line))


# only take the question and the chosen answer
orca_test_data = [{"question": data["prompt"], "answer": data["chosen"]} for data in orca_test_data]

# WebGPT comparisons
#? https://huggingface.co/datasets/openai/webgpt_comparisons
web_gpt_test_data = []
with open("model/datasets/dpo_webgpt_comparaisons_test.jsonl", "r") as file:
    for line in file:
        web_gpt_test_data.append(json.loads(line))

# only take the question and the chosen answer
web_gpt_test_data = [{"question": data["prompt"], "answer": data["chosen"]} for data in web_gpt_test_data]


# Stanford Human Preferences Dataset (SHP)
#? https://huggingface.co/datasets/stanfordnlp/SHP
shp_test_data = []
with open("model/datasets/dpo_shp_test.jsonl", "r") as file:
    for line in file:
        line = line.strip()
        data = json.loads(line)
        shp_test_data.append(data)
    
# only take the question and the chosen answer
shp_test_data = [{"question": data["prompt"], "answer": data["chosen"]} for data in shp_test_data]


# stemQ dataset
stem_test_data = []
with open("model/datasets/stem_test.jsonl", "r") as file:
    for line in file:
        line = line.strip()
        data = json.loads(line)
        stem_test_data.append(data)

# only take the question and the chosen answer
stem_test_data = [{"question": data["question"], "answer": data["answer"]} for data in stem_test_data]

# ===========================================
# Create the full test dataset with all parts
# ===========================================

full_test_data = orca_test_data + web_gpt_test_data + shp_test_data + stem_test_data

# ===========================================
# Remove duplicates for the questions
# If a question appears several times, there will be more than one answer.
# The new dataset has 'question' (unique) and then a list of 'answers'.
# ===========================================

dataset = full_test_data

question_answers_dict = {}
counter = 0

for item in dataset:
    question = item['question']
    answer = item['answer']
    
    if question in question_answers_dict:
        counter += 1
        question_answers_dict[question].append(answer)
    else:
        question_answers_dict[question] = [answer]

# ===========================================
# Save the final test dataset
# ===========================================

with open('model/datasets/full_test_dataset.jsonl', 'w') as file:
    for question, answers in question_answers_dict.items():
        entry = {'question': question, 'answers': answers}
        file.write(json.dumps(entry) + '\n')

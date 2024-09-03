from transformers import T5ForConditionalGeneration, T5Tokenizer
import fine_tuning_helper as helper
import json
import os
import evaluate

print('-----START MODEL PERFORMANCE TEST-----')

# ===========================================
# Load the test datasets 
# ===========================================

test_data = []
with open("model/datasets/full_test_dataset.jsonl", "r") as file:
    for line in file:
        test_data.append(json.loads(line))

# ===========================================
# Load the model and tokenizer and create the prompts
# ===========================================

model_path = "models/flan-t5-large"
model_name = "flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# split into questions and answers
questions, answers = helper.split_test_dataset(test_data)

# create the prompting inputs
prompting_inputs, labels = helper.create_prompting_inputs_test_dataset(questions, answers, tokenizer)

# ===========================================
# Answer generation 
# ===========================================

print('START PREDICTION GENERATION')

model.eval()
predictions = []
for i in range(len(prompting_inputs)):
    input_text = prompting_inputs[i]
    input_ids = tokenizer(input_text, truncation=True, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    # outputs[0] -> contains the generated answer 
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predictions.append(prediction)

print('END PREDICTION GENERATION')

# ===========================================
# Useful functions
# ===========================================

def normalize_text(input_string):
    """
    Removing articles, punctuations and standardizing whitespaces.

    Args:
        input_string: input string

    Returns:
        normalized_string: normalized string
    """
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    normalized_string = white_space_fix(remove_articles(remove_punc(lower(input_string))))

    return normalized_string


def exact_match(prediction, label):
    """
    Function to calculate the exact match between the prediction and the label.

    Args:
        prediction: predicted answer by the model
        label: correct answer

    Returns:
        exact_match: 1 if the prediction is correct, 0 otherwise
    """
    normalized_prediction = normalize_text(prediction)
    normalized_label = normalize_text(label)

    EM = int(normalized_prediction == normalized_label)

    return EM

def f1_score(prediction, label): 
    """
    Function to calculate the F1 score between the prediction and the label.

    Args:
        prediction: predicted answer by the model
        label: correct answer

    Returns:
        f1_score: F1 score between the prediction and the label
    """
    predicted_tokens = normalize_text(prediction).split()
    label_tokens = normalize_text(label).split()
    
    # either the prediction or the label is no-answer -> f1 = 1 if they agree, 0 otherwise
    if len(predicted_tokens) == 0 or len(label_tokens) == 0:
        return int(predicted_tokens == label_tokens)
    
    common_tokens = set(predicted_tokens) & set(label_tokens)
    
    # no common tokens -> f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    precision = len(common_tokens) / len(predicted_tokens)
    recall = len(common_tokens) / len(label_tokens)

    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return f1_score

# ===========================================
# Calculate the scores -> EM and F1 
# ===========================================

print('START EM AND F1')

exact_matches = []
f1_scores = []

for i in range(len(predictions)):
    # take the highest value out of the possible answers for both exact match and f1 score
    exact_match_value = max([exact_match(predictions[i], label) for label in labels[i]])
    f1_score_value = max([f1_score(predictions[i], label) for label in labels[i]])
    exact_matches.append(exact_match_value)
    f1_scores.append(f1_score_value)

# store the EM results 
file_path_EM = "model_perf_results/EM_results.json-" + model_name
if os.path.exists(file_path_EM):
    with open(file_path_EM, "w") as file:
        json.dump(exact_matches, file)
else:
    with open(file_path_EM, "x") as file:
        json.dump(exact_matches, file)

# store the F1 results 
file_path_f1 = "model_perf_results/F1_results.json-" + model_name
if os.path.exists(file_path_f1):
    with open(file_path_f1, "w") as file:
        json.dump(f1_scores, file)
else:
    with open(file_path_f1, "x") as file:
        json.dump(f1_scores, file)

print('END EM AND F1')

# ===========================================
# Calculate the scores -> BLEU
#? https://huggingface.co/spaces/evaluate-metric/bleu
# =========================================== 

print('START BLEU')

bleu = evaluate.load('bleu')
list_predictions = []
list_references = []
list_labels = []
for i in range(len(predictions)):
    list_predictions.append([predictions[i]])
    list_labels = []
    for label in labels[i]:
        list_labels.append([label])
    list_references.append(list_labels)
bleu_results = bleu.compute(predictions=predictions, references=labels)

# store the BLEU results 
file_path_bleu = "model_perf_results/bleu_results.json-" + model_name
if os.path.exists(file_path_bleu):
    with open(file_path_bleu, "w") as file:
        json.dump(bleu_results, file)
else:
    with open(file_path_bleu, "x") as file:
        json.dump(bleu_results, file)

print('END BLEU')

# ===========================================
# Calculate the scores -> ROUGE
#? https://huggingface.co/spaces/evaluate-metric/rouge
# =========================================== 

print('START ROUGE')

rouge = evaluate.load('rouge')
rouge_results = rouge.compute(predictions=predictions, references=labels)

# store the ROUGE results
file_path_rouge = "model_perf_results/rouge_results.json-" + model_name
if os.path.exists(file_path_rouge):
    with open(file_path_rouge, "w") as file:
        json.dump(rouge_results, file)
else:
    with open(file_path_rouge, "x") as file:
        json.dump(rouge_results, file)

print('END ROUGE')

# ===========================================
# Calculate the scores -> BLEURT
#? https://huggingface.co/spaces/evaluate-metric/bleurt
# =========================================== 

print('START BLEURT')

bleurt = evaluate.load('bleurt', module_type='metric')
bleurt_results = []
for i in range(len(predictions)):
    label_bleurt_scores = []
    for l in range(len(labels[i])):
        label_bleurt_score = bleurt.compute(predictions=[predictions[i]], references=[labels[i][l]])
        label_bleurt_scores.append(label_bleurt_score["scores"][0])
    bleurt_score = max(label_bleurt_scores)
    bleurt_results.append(bleurt_score)

# store the BLEURT results
file_path_bleurt = "model_perf_results/bleurt_results.json-" + model_name
if os.path.exists(file_path_bleurt):
    with open(file_path_bleurt, "w") as file:
        json.dump(bleurt_results, file)
else:
    with open(file_path_bleurt, "x") as file:
        json.dump(bleurt_results, file)

print('END BLEURT')

print('-----END MODEL PERFORMANCE TEST-----')

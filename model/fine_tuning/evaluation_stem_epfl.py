from transformers import T5ForConditionalGeneration, T5Tokenizer
import json
import fine_tuning_helper as helper
import evaluate
import os

# ===========================================
# Load model, tokenizer and test dataset
# ===========================================

model_path = "/home/ckalberm/project-m2-2024-chatbots-r-us/models/flan-t5-base-stem-epfl-exams"
model_name = "flan-t5-base-stem-epfl-exams"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# load test dataset -> EPFL and STEM datasets
# extract the questions, answers and courses
with open("data_fine_tuning/epfl_test_dataset.json", "r") as file:
    test_data_epfl = json.load(file)
with open("data_fine_tuning/stem_test_dataset.json", "r") as file:
    test_data_stem = json.load(file)

# split into questions, answers and courses
questions_epfl, answers_epfl, courses_epfl = helper.epfl_course_split_dataset(test_data_epfl)
questions_stem, answers_stem, courses_stem = helper.stem_split_dataset(test_data_stem)

# combine both datasets
questions = questions_epfl + questions_stem
answers = answers_epfl + answers_stem
courses = courses_epfl + courses_stem

# create the prompting inputs
prompting_inputs, labels = helper.create_prompting_inputs(questions, courses, answers, tokenizer)

# ===========================================
# Answer generation 
# ===========================================

model.eval()
# generate the answers
predictions = []
for i in range(len(prompting_inputs)):
    input_text = prompting_inputs[i]
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    # outputs[0] -> contains the generated answer 
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predictions.append(prediction)

# ===========================================
# Calculate the scores -> BLEU
#? https://huggingface.co/spaces/evaluate-metric/bleu
# =========================================== 

bleu = evaluate.load('bleu')
bleu_results = bleu.compute(predictions=predictions, references=labels)
# store the BLEU results 
file_path_bleu = "evaluation_results/bleu_results.json-" + model_name
if os.path.exists(file_path_bleu):
    with open(file_path_bleu, "w") as file:
        json.dump(bleu_results, file)
else:
    with open(file_path_bleu, "x") as file:
        json.dump(bleu_results, file)

# ===========================================
# Calculate the scores -> ROUGE
#? https://huggingface.co/spaces/evaluate-metric/rouge
# =========================================== 

rouge = evaluate.load('rouge')
rouge_results = rouge.compute(predictions=predictions, references=labels)
# store the ROUGE results
file_path_rouge = "evaluation_results/rouge_results.json-" + model_name
if os.path.exists(file_path_rouge):
    with open(file_path_rouge, "w") as file:
        json.dump(rouge_results, file)
else:
    with open(file_path_rouge, "x") as file:
        json.dump(rouge_results, file)

# ===========================================
# Calculate the scores -> BLEURT
#? https://huggingface.co/spaces/evaluate-metric/bleurt
# =========================================== 

bleurt = evaluate.load('bleurt', module_type='metric')
bleurt_results = bleurt.compute(predictions=predictions, references=labels)
# store the BLEURT results
file_path_bleurt = "evaluation_results/bleurt_results.json-" + model_name
if os.path.exists(file_path_bleurt):
    with open(file_path_bleurt, "w") as file:
        json.dump(bleurt_results, file)
else:
    with open(file_path_bleurt, "x") as file:
        json.dump(bleurt_results, file)

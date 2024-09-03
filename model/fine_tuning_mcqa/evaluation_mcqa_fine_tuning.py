from transformers import T5ForConditionalGeneration, T5Tokenizer
import json
import fine_tuning_helper as helper
import evaluate
import os

# ===========================================
# Load model, tokenizer and test dataset
# ===========================================

model_path = "/home/ckalberm/project-m3-2024-chatbots-r-us/models/flan-t5-large-LoRA-merged-mcqa-ai2-mcqa-sciq"
model_name = "flan-t5-large-LoRA-merged-mcqa-ai2-mcqa-sciq"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# load the test datasets -> ai2 and sciq
dataset_test_ai2 = helper.read_jsonl_file("/home/ckalberm/project-m3-2024-chatbots-r-us/model/datasets/mcqa/test/mcqa_ai2_arc_test.jsonl")
dataset_test_sciq = helper.read_jsonl_file("/home/ckalberm/project-m3-2024-chatbots-r-us/model/datasets/mcqa/test/mcqa_sciq_test.jsonl")
dataset_test_mmlu = helper.read_jsonl_file("/home/ckalberm/project-m3-2024-chatbots-r-us/model/datasets/mcqa/test/mcqa_mmlu_test.jsonl")

# get questions and answers from the datasets
questions_ai2, answers_ai2, subjects_ai2 = helper.mcqa_split_dataset(dataset_test_ai2)
questions_sciq, answers_sciq, subjects_sciq = helper.mcqa_split_dataset(dataset_test_sciq)
questions_mmlu, answers_mmlu, subjects_mmlu = helper.mcqa_split_dataset(dataset_test_mmlu)

# combine all the datasets
questions = questions_ai2 + questions_sciq + questions_mmlu
answers = answers_ai2 + answers_sciq + answers_mmlu
subjects = subjects_ai2 + subjects_sciq + subjects_mmlu

# create the prompting inputs
prompting_inputs, labels = helper.mcqa_create_prompting_inputs(questions, answers)

# ===========================================
# Answer generation 
# ===========================================

print('START PREDICTION')

model.eval()
# generate the answers
predictions = []
for i in range(len(prompting_inputs)):
    print('Prediction:', i, 'of', len(prompting_inputs))
    input_text = prompting_inputs[i]
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    # outputs[0] -> contains the generated answer 
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predictions.append(prediction)

# save the predictions and labels in a file
    file_path_pred = "/home/ckalberm/project-m3-2024-chatbots-r-us/model/fine-tuning/predictions/" + model_name
    if os.path.exists(file_path_pred):
        with open(file_path_pred, "w") as file:
            json.dump(predictions, file)
    else:
        with open(file_path_pred, "x") as file:
            json.dump(predictions, file)

print('END PREDICTION')

# ===========================================
# Calculate the scores -> F1 score
#? https://huggingface.co/spaces/evaluate-metric/f1/blob/main/f1.py
#? https://huggingface.co/docs/evaluate/transformers_integrations
# =========================================== 

print('START F1 SCORE')

# encode the predictions and the labels -> should be a list of integers
predictions_encoded = tokenizer(predictions, padding=True, truncation=True, return_tensors="pt").input_ids
predictions_list = predictions_encoded[:, 0].tolist()
labels_encoded = tokenizer(labels, padding=True, truncation=True, return_tensors="pt").input_ids
labels_list = labels_encoded[:, 0].tolist()

f1_score = evaluate.load('f1')
f1_results = f1_score.compute(predictions=predictions_list, references=labels_list, average='weighted')
# store the F1 results
file_path_f1 = "/home/ckalberm/project-m3-2024-chatbots-r-us/model/fine-tuning/mcqa_evaluation_results/" + model_name
if os.path.exists(file_path_f1):
    with open(file_path_f1, "w") as file:
        json.dump(f1_results, file)
else:
    with open(file_path_f1, "x") as file:
        json.dump(f1_results, file)

print('END F1 SCORE')
from transformers import T5ForConditionalGeneration, T5Tokenizer
import json
import fine_tuning_helper as helper
import evaluate
import os

# ===========================================
# Load model, tokenizer and test dataset
# ===========================================

model_path = "/home/ckalberm/project-m3-2024-chatbots-r-us/models/flan-t5-large_mcqa-ai2-sciq-M1-LoRA-merged"
model_name = "flan-t5-large-LoRA-merged-mcqa-ai2-sciq-M1"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# load the test datasets -> ai2 and sciq
dataset_test_ai2 = helper.read_jsonl_file("/home/ckalberm/project-m3-2024-chatbots-r-us/model/datasets/mcqa/test/mcqa_ai2_arc_test.jsonl")
dataset_test_sciq = helper.read_jsonl_file("/home/ckalberm/project-m3-2024-chatbots-r-us/model/datasets/mcqa/test/mcqa_sciq_test.jsonl")
dataset_test_mmlu = helper.read_jsonl_file("/home/ckalberm/project-m3-2024-chatbots-r-us/model/datasets/mcqa/test/mcqa_mmlu_test.jsonl")
dataset_test_M1 = helper.read_jsonl_file("/home/ckalberm/project-m3-2024-chatbots-r-us/model/datasets/mcqa/test/mcqa_M1_test.jsonl")

# get questions and answers from the individual datasets
questions_ai2, answers_ai2, subjects_ai2 = helper.mcqa_split_dataset(dataset_test_ai2)
questions_sciq, answers_sciq, subjects_sciq = helper.mcqa_split_dataset(dataset_test_sciq)
questions_mmlu, answers_mmlu, subjects_mmlu = helper.mcqa_split_dataset(dataset_test_mmlu)
questions_M1, answers_M1 = helper.mcqa_split_dataset(dataset_test_M1)

# create a combined dataset
questions_combined = questions_ai2 + questions_sciq + questions_mmlu + questions_M1
answers_combined = answers_ai2 + answers_sciq + answers_mmlu + answers_M1

# create the prompting inputs for the datasets
prompting_inputs_ai2, labels_ai2 = helper.mcqa_create_prompting_inputs(questions_ai2, answers_ai2)
prompting_inputs_sciq, labels_sciq = helper.mcqa_create_prompting_inputs(questions_sciq, answers_sciq)
prompting_inputs_mmlu, labels_mmlu = helper.mcqa_create_prompting_inputs(questions_mmlu, answers_mmlu)
prompting_inputs_M1, labels_M1 = helper.mcqa_create_prompting_inputs(questions_M1, answers_M1)
promting_inputs_combined, labels_combined = helper.mcqa_create_prompting_inputs(questions_combined, answers_combined)

# ===========================================
# Answer generation 
# ===========================================

def generate_predictions(dataset_name, prompting_inputs):
    """
    Generate predictions (answers) for the given prompting inputs

    Args:
        prompting_inputs: list of prompting inputs

    Returns:
        predictions: list of generated predictions (answers)
    """
    print('START PREDICTION FOR: ', dataset_name, ' DATASET')
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
    print('END PREDICTION FOR: ', dataset_name, ' DATASET')

    return predictions


def encode_predictions_and_labels(predictions, labels):
    """
    Encode the predictions and the labels -> should be a list of integers

    Args:
        predictions: all the predictions
        labels: all the labels

    Returns:
        predictions_encoded: list of encoded predictions
        labels_encoded: list of encoded labels
    """
    predictions_encoded = tokenizer(predictions, padding=True, truncation=True, return_tensors="pt").input_ids
    predictions_list = predictions_encoded[:, 0].tolist()
    labels_encoded = tokenizer(labels, padding=True, truncation=True, return_tensors="pt").input_ids
    labels_list = labels_encoded[:, 0].tolist()

    return predictions_list, labels_list


# get the predictions
predictions_ai2 = generate_predictions('ai2', prompting_inputs_ai2)
predictions_sciq = generate_predictions('sciq', prompting_inputs_sciq)
predictions_mmlu = generate_predictions('mmlu', prompting_inputs_mmlu)
predictions_M1 = generate_predictions('M1', prompting_inputs_M1)
predictions_combined = predictions_ai2 + predictions_sciq + predictions_mmlu + predictions_M1
#predictions_combined = generate_predictions('combined', promting_inputs_combined)

# encode the predictions and the labels 
predictions_list_ai2, labels_list_ai2 = encode_predictions_and_labels(predictions_ai2, labels_ai2)
predictions_list_sciq, labels_list_sciq = encode_predictions_and_labels(predictions_sciq, labels_sciq)
predictions_list_mmlu, labels_list_mmlu = encode_predictions_and_labels(predictions_mmlu, labels_mmlu)
predictions_list_M1, labels_list_M1 = encode_predictions_and_labels(predictions_M1, labels_M1)
predictions_list_combined, labels_list_combined = encode_predictions_and_labels(predictions_combined, labels_combined)

# ===========================================
# Calculate the scores -> F1 score
#? https://huggingface.co/spaces/evaluate-metric/f1/blob/main/f1.py
#? https://huggingface.co/docs/evaluate/transformers_integrations
# =========================================== 

print('START F1 SCORE')

# load the metric
f1_score = evaluate.load('f1')

# calculate the F1 scores
f1_results_ai2 = f1_score.compute(predictions=predictions_list_ai2, references=labels_list_ai2, average='weighted')
f1_results_sciq = f1_score.compute(predictions=predictions_list_sciq, references=labels_list_sciq, average='weighted')
f1_results_mmlu = f1_score.compute(predictions=predictions_list_mmlu, references=labels_list_mmlu, average='weighted')
f1_results_M1 = f1_score.compute(predictions=predictions_list_M1, references=labels_list_M1, average='weighted')
f1_results_combined = f1_score.compute(predictions=predictions_list_combined, references=labels_list_combined, average='weighted')

# store the F1 results
file_path_f1 = "/home/ckalberm/project-m3-2024-chatbots-r-us/model/fine-tuning/mcqa_evaluation_results/F1_scores/" + model_name
if os.path.exists(file_path_f1):
    with open(file_path_f1, "w") as file:
        json.dump({'combined': f1_results_combined, 'ai2': f1_results_ai2, 'sciq': f1_results_sciq, 'mmlu': f1_results_mmlu, 'M1': f1_results_M1}, file)
else:
    with open(file_path_f1, "x") as file:
        json.dump({'combined': f1_results_combined, 'ai2': f1_results_ai2, 'sciq': f1_results_sciq, 'mmlu': f1_results_mmlu, 'M1': f1_results_M1}, file)

print('END F1 SCORE')

# ===========================================
# Calculate the scores -> accuracy
#? https://huggingface.co/spaces/evaluate-metric/accuracy/blob/main/accuracy.py
#? https://huggingface.co/docs/evaluate/transformers_integrations
# =========================================== 

print('START ACCURACY')

# load the metric
accuracy = evaluate.load('accuracy')

# calculate the accuracy
accuracy_results_ai2 = accuracy.compute(predictions=predictions_list_ai2, references=labels_list_ai2)
accuracy_results_sciq = accuracy.compute(predictions=predictions_list_sciq, references=labels_list_sciq)
accuracy_results_mmlu = accuracy.compute(predictions=predictions_list_mmlu, references=labels_list_mmlu)
accuracy_results_M1 = accuracy.compute(predictions=predictions_list_M1, references=labels_list_M1)
accuracy_results_combined = accuracy.compute(predictions=predictions_list_combined, references=labels_list_combined)

# store the accuracy results
file_path_accuracy = "/home/ckalberm/project-m3-2024-chatbots-r-us/model/fine-tuning/mcqa_evaluation_results/accuracies/" + model_name
if os.path.exists(file_path_accuracy):
    with open(file_path_accuracy, "w") as file:
        json.dump({'combined': accuracy_results_combined, 'ai2': accuracy_results_ai2, 'sciq': accuracy_results_sciq, 'mmlu': accuracy_results_mmlu, 'M1': accuracy_results_M1}, file)
else:
    with open(file_path_accuracy, "x") as file:
        json.dump({'combined': accuracy_results_combined, 'ai2': accuracy_results_ai2, 'sciq': accuracy_results_sciq, 'mmlu': accuracy_results_mmlu, 'M1': accuracy_results_M1}, file)

print('END ACCURACY')



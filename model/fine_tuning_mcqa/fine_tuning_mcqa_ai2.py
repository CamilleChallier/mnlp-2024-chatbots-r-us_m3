from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import torch 
from datasets import Dataset
import evaluate
import fine_tuning_helper as helper  
import os

# ===========================================
# Load the model and the tokenizer
# ===========================================

#model_checkpoint = "/home/ckalberm/project-m3-2024-chatbots-r-us/models/flan-t5-large_finetuned_LoRA_merged_OK"
model_checkpoint = "/home/ckalberm/project-m3-2024-chatbots-r-us/models/flan-t5-large-LoRA-merged-mcqa-sciq"
#model_name = "flan-t5-large-LoRA-merged"
model_name = "flan-t5-large-LoRA-merged-mcqa-sciq"
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_checkpoint)

# ===========================================
# Create the dataset -> ai2
# ===========================================

# load the train and validation datasets from the jsonl file
dataset_train_ai2 = helper.read_jsonl_file("/home/ckalberm/project-m3-2024-chatbots-r-us/model/datasets/mcqa/train/mcqa_ai2_arc_train.jsonl")
dataset_evaluation_ai2 = helper.read_jsonl_file("/home/ckalberm/project-m3-2024-chatbots-r-us/model/datasets/mcqa/evaluation/mcqa_ai2_arc_validation.jsonl")

# get questions and answers from the datasets
questions_train, answers_train, subjects_train = helper.mcqa_split_dataset(dataset_train_ai2)
questions_val, answers_val, subjects_val = helper.mcqa_split_dataset(dataset_evaluation_ai2)

# create the prompting inputs
prompting_inputs_train, labels_train = helper.mcqa_create_prompting_inputs(questions_train, answers_train)
prompting_inputs_val, labels_val = helper.mcqa_create_prompting_inputs(questions_val, answers_val)

# encode the prompting inputs and the answers
# labels -> contains 0 for padding tokens and 1 for the rest, so the padding tokens are already masked 
# no need to transform into -100
#? train_data_sciq :  dict_keys(['input_ids', 'attention_mask', 'labels'])
train_data_ai2 = tokenizer(text=prompting_inputs_train, text_target=labels_train, padding=True, truncation=True)
#? val_data_sciq :  dict_keys(['input_ids', 'attention_mask', 'labels'])
val_data_ai2 = tokenizer(text=prompting_inputs_val, text_target=labels_val, padding=True, truncation=True)

# create the datasets
train_dataset_ai2 = Dataset.from_dict({'input_ids': train_data_ai2['input_ids'], 'attention_mask': train_data_ai2['attention_mask'], 'labels': train_data_ai2['labels']})
val_dataset_ai2 = Dataset.from_dict({'input_ids': val_data_ai2['input_ids'], 'attention_mask': val_data_ai2['attention_mask'], 'labels': val_data_ai2['labels']})

# ===========================================
# Set the training arguments, create a trainer object 
#? https://huggingface.co/docs/transformers/main_classes/trainer
# ===========================================

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
batch_size = 2

# define the function to compute the metrics
#? https://huggingface.co/spaces/evaluate-metric/f1/blob/main/f1.py
metrics = evaluate.load('f1')

def compute_metrics(eval_pred):
    """
    Function to compute the metrics for the evaluation.

    Args:
        eval_pred: tuple (predictions, labels)

    Returns:
        metrics: dictionary with the computed metrics
    """
    pred = eval_pred.predictions
    lab = eval_pred.label_ids

    # take the second element to get the prediction, first element always 0, then prediction, then 1, then 0 again
    predictions = pred[:, 1]
    # take the first element to get the labels, first element always prediction, then 1, then 0
    labels = lab[:, 0]
    # convert to tensors 
    predictions = torch.tensor(predictions)
    labels = torch.tensor(labels)
    return metrics.compute(predictions=predictions, references=labels, average='weighted')

    
#? https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/trainer#transformers.Seq2SeqTrainingArguments
training_args = Seq2SeqTrainingArguments(
    output_dir=f"{model_name}-mcqa-ai2",
    report_to="tensorboard",
    logging_dir="./tensorboard/"+model_name+"-mcqa-ai2",
    evaluation_strategy = "epoch",
    num_train_epochs = 4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    # log (show) losses at each batch
    logging_steps=1,
    learning_rate=2e-5,
    # how many checkpoints we want to save
    save_total_limit=3,
    weight_decay=0.01,
    # use the generate method to evaluate the model
    predict_with_generate=True,)

#? https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/trainer#transformers.Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_ai2,
    eval_dataset=val_dataset_ai2,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,)

# ===========================================
# Fine-tuning and saving the model and tokenizer
# ===========================================

trainer.train()
# save the model
trainer.save_model(f"models/{model_name}-mcqa-ai2")
# save the tokenizer
tokenizer.save_pretrained(f"models/{model_name}-mcqa-ai2")

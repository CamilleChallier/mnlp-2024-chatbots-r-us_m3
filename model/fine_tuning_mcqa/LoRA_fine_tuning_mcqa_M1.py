from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import torch 
from datasets import Dataset
import evaluate
import fine_tuning_helper as helper  
from peft import LoraConfig, get_peft_model, TaskType

# ===========================================
# Load the model and the tokenizer
# ===========================================

model_checkpoint = "/home/ckalberm/project-m3-2024-chatbots-r-us/models/flan-t5-large_mcqa-ai2-sciq-LoRA-merged"
model_name = "flan-t5-large-LoRA-merged-mcqa-ai2-sciq"
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_checkpoint)

# ===========================================
# Create the LoRA model
#? https://huggingface.co/docs/peft/package_reference/lora
# ===========================================

peft_config = LoraConfig(
    # rank
    r=16, 
    lora_alpha=32,
    target_modules=['q', 'v'],
    lora_dropout=0.05,
    bias="none",
    # flan-T5
    task_type=TaskType.SEQ_2_SEQ_LM 
)

lora_model = get_peft_model(model, peft_config)

# ===========================================
# Create the dataset -> M1
# ===========================================

# load the train and validation datasets from the jsonl file
dataset_train_M1 = helper.read_jsonl_file("/home/ckalberm/project-m3-2024-chatbots-r-us/model/datasets/mcqa/train/mcqa_M1_train.jsonl")
dataset_evaluation_M1 = helper.read_jsonl_file("/home/ckalberm/project-m3-2024-chatbots-r-us/model/datasets/mcqa/evaluation/mcqa_M1_evaluation.jsonl")

# get questions and answers from the datasets
questions_train, answers_train = helper.mcqa_split_dataset(dataset_train_M1)
questions_val, answers_val = helper.mcqa_split_dataset(dataset_evaluation_M1)

# create the prompting inputs
prompting_inputs_train, labels_train = helper.mcqa_create_prompting_inputs(questions_train, answers_train)
prompting_inputs_val, labels_val = helper.mcqa_create_prompting_inputs(questions_val, answers_val)

# encode the prompting inputs and the answers
# labels -> contains 0 for padding tokens and 1 for the rest, so the padding tokens are already masked 
# no need to transform into -100
#? train_data_M1 :  dict_keys(['input_ids', 'attention_mask', 'labels'])
train_data_M1 = tokenizer(text=prompting_inputs_train, text_target=labels_train, padding=True, truncation=True)
#? val_data_M1 :  dict_keys(['input_ids', 'attention_mask', 'labels'])
val_data_M1 = tokenizer(text=prompting_inputs_val, text_target=labels_val, padding=True, truncation=True)

# create the datasets
train_dataset_M1 = Dataset.from_dict({'input_ids': train_data_M1['input_ids'], 'attention_mask': train_data_M1['attention_mask'], 'labels': train_data_M1['labels']})
val_dataset_M1 = Dataset.from_dict({'input_ids': val_data_M1['input_ids'], 'attention_mask': val_data_M1['attention_mask'], 'labels': val_data_M1['labels']})

# ===========================================
# Define the metrics -> F1 score and accuracy
#? https://huggingface.co/spaces/evaluate-metric/f1/blob/main/f1.py
#? https://huggingface.co/spaces/evaluate-metric/accuracy/blob/main/accuracy.py
# ===========================================

def compute_metrics(eval_pred):
    """
    Function to compute the metrics for the evaluation.

    Args:
        eval_pred: tuple (predictions, labels)

    Returns:
        metrics: dictionary with the computed metrics
    """
    # load the metrics
    f1_score = evaluate.load('f1')
    accuracy = evaluate.load('accuracy')

    pred = eval_pred.predictions
    lab = eval_pred.label_ids

    # take the second element to get the prediction, first element always 0, then prediction, then 1, then 0 again
    predictions = pred[:, 1]
    # take the first element to get the labels, first element always prediction, then 1, then 0
    labels = lab[:, 0]
    # convert to tensors 
    predictions = torch.tensor(predictions)
    labels = torch.tensor(labels)

    f1_score_result = f1_score.compute(predictions=predictions, references=labels, average='weighted')
    accuracy_result = accuracy.compute(predictions=predictions, references=labels)

    return {"f1_score": f1_score_result, "accuracy": accuracy_result}

# ===========================================
# Set the training arguments, create a trainer object 
#? https://huggingface.co/docs/transformers/main_classes/trainer
# ===========================================

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
batch_size = 2    

#? https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/trainer#transformers.Seq2SeqTrainingArguments
training_args = Seq2SeqTrainingArguments(
    output_dir=f"{model_name}-M1-LoRA",
    report_to="tensorboard",
    logging_dir="./tensorboard/"+model_name+"-M1-LoRA",
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
    #model=model,
    model=lora_model,
    args=training_args,
    train_dataset=train_dataset_M1,
    eval_dataset=val_dataset_M1,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,)

# ===========================================
# Fine-tuning and saving the model and tokenizer
# ===========================================

trainer.train()
# save the model
trainer.save_model(f"models/{model_name}-M1-LoRA")
# save the tokenizer
tokenizer.save_pretrained(f"models/{model_name}-M1-LoRA")

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import torch 
from datasets import Dataset
from sklearn.model_selection import train_test_split
import evaluate
import json
import fine_tuning_helper as helper 

# ===========================================
# Load the model and the tokenizer
# ===========================================

#model_checkpoint = "google/flan-t5-base"
model_checkpoint = "/home/ckalberm/project-m2-2024-chatbots-r-us/models/flan-t5-base-stem"
#model_name = 'flan-t5-base'
model_name = 'flan-t5-base-stem'
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_checkpoint)

# ===========================================
# Create the dataset -> EPFL exams
# ===========================================

# load the train and validation datasets from the json file
with open("data_fine_tuning/epfl_train_dataset.json", "r") as file:
    dataset_train_epfl_exams = json.load(file)
with open("data_fine_tuning/epfl_val_dataset.json", "r") as file:
    dataset_val_epfl_exams = json.load(file)

# get questions and answers from the datasets
questions_train, answers_train, courses_train = helper.epfl_course_split_dataset(dataset_train_epfl_exams)
questions_val, answers_val, courses_val = helper.epfl_course_split_dataset(dataset_val_epfl_exams)

# create the prompting inputs
prompting_inputs_train, labels_train = helper.create_prompting_inputs(questions_train, courses_train, answers_train, tokenizer)
prompting_inputs_val, labels_val = helper.create_prompting_inputs(questions_val, courses_val, answers_val, tokenizer)

# encode the prompting inputs and the answers
# labels -> contains 0 for padding tokens and 1 for the rest, so the padding tokens are already masked 
# no need to transform into -100
#? train_data_epfl :  dict_keys(['input_ids', 'attention_mask', 'labels'])
train_data_epfl = tokenizer(text=prompting_inputs_train, text_target=labels_train, padding=True, truncation=True)
#? val_data_epfl :  dict_keys(['input_ids', 'attention_mask', 'labels'])
val_data_epfl = tokenizer(text=prompting_inputs_val, text_target=labels_val, padding=True, truncation=True)

# create the datasets
train_dataset_epfl = Dataset.from_dict({'input_ids': train_data_epfl['input_ids'], 'attention_mask': train_data_epfl['attention_mask'], 'labels': train_data_epfl['labels']})
val_dataset_epfl = Dataset.from_dict({'input_ids': val_data_epfl['input_ids'], 'attention_mask': val_data_epfl['attention_mask'], 'labels': val_data_epfl['labels']})

# ===========================================
# Set the training arguments, create a trainer object 
# ===========================================

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
batch_size = 2

training_args = Seq2SeqTrainingArguments(
    output_dir=f"{model_name}-epfl-exams",
    report_to="tensorboard",
    logging_dir="./tensorboard/"+model_name+"-epfl-exams",
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

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_epfl,
    eval_dataset=val_dataset_epfl,
    tokenizer=tokenizer,
    data_collator=data_collator,)

# ===========================================
# Fine-tuning and saving the model and tokenizer
# ===========================================

trainer.train()
# save the model
trainer.save_model(f"models/{model_name}-epfl-exams")
# save the tokenizer
tokenizer.save_pretrained(f"models/{model_name}-epfl-exams")


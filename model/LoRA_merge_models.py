from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

# ===========================================
# Merge the base model with the PEFT model
# ===========================================

#base_model_path = "/home/ckalberm/project-m2-2024-chatbots-r-us/models/flan-t5-large"
base_model_path = "/home/ckalberm/project-m3-2024-chatbots-r-us/models/flan-t5-large_mcqa-ai2-sciq-LoRA-merged"
peft_model_path = "/home/ckalberm/project-m3-2024-chatbots-r-us/models/flan-t5-large-LoRA-merged-mcqa-ai2-sciq-M1-LoRA"
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_path)
model = PeftModel.from_pretrained(base_model, peft_model_path)
tokenizer = AutoTokenizer.from_pretrained(peft_model_path)
merged_model = model.merge_and_unload()

# ===========================================
# Save the merged model for later use
# ===========================================

# save the merged model
merged_model_path = "models/flan-t5-large_mcqa-ai2-sciq-M1-LoRA-merged"
merged_model.save_pretrained(merged_model_path)
# save the tokenizer
tokenizer.save_pretrained(merged_model_path)



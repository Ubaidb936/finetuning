# IMPORTS
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import datasets
from datasets import load_dataset
from trl import SFTTrainer
from peft import PeftConfig, PeftModel
from multiprocessing import cpu_count
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb
import transformers
import config 

hf_token = config.hf_token
model_id = config.model_id
dataset_name = config.dataset_name
adapter_hub_name = config.adapter_hub_name

if hf_token is None:
    raise ValueError("hf_token is None. Please set the huggingFace token in config.py.")

if model_id is None:
    raise ValueError("model_id is None. Please set the model_id in config.py.")

if dataset_name is None:
    raise ValueError("dataset_name is None. Please set the dataset_name in config.py.")

if adapter_hub_name is None:
    raise ValueError("adapter_hub_name is None. Please set the adapter_hub_name in config.py.")




# LOGIN TO HUB FOR MODEL DEPLOYMENT
from huggingface_hub import login
login(hf_token)

# LOADING THE TOKENIZER
print("loading tokenizer.........................")
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)


print("loading dataset..........")
# LOAD DATA FROM HUGGINFACE
data = load_dataset(dataset_name, split='train')

# PREPARE DATA FOR FINE-TUNING
def generate_prompt(data_point):
    
    if data_point['context']:
        prompt_template = """
        <|system|>
        Answer the question based on your knowledge. Use the following context to help:
        {context}
    
        </s>
        <|user|>
        {question}
        </s>
        <|assistant|>
        {answer}
        </s>
        """
        prompt = prompt_template.format(
        context=data_point["context"], 
        question=data_point["question"], 
        answer=data_point["answer"]
    )
        

    # Without context
    else:
        prompt_template = """
        <|system|>
        Answer the question based on your knowledge.
        </s>
        <|user|>
        {question}
        </s>
        <|assistant|>
        {answer}
        </s>
        """
        prompt = prompt_template.format(
        question=data_point["question"], 
        answer=data_point["answer"]
        )
    return prompt

print("Preparing dataset for fine-tuning................")
prompt = [generate_prompt(data_point) for data_point in data]
data = data.add_column("prompt", prompt);
data = data.map(lambda sample: tokenizer(sample["prompt"]),num_proc=cpu_count(), batched=True)
# data = data.remove_columns(['Context', 'Response'])
data = data.shuffle(seed=1234)
data = data.train_test_split(test_size=0.01)
train_data = data["train"]
test_data = data["test"]


# LOADING MODEL IN N(4, 8.....) BIT
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

d_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None

print("loading model.......................")
model = AutoModelForCausalLM.from_pretrained(
  model_id,
  quantization_config=bnb_config,
  device_map=d_map,
 
)

##Loading the layers to finetune
def find_all_linear_names(model):
  cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
  lora_module_names = set()
  for name, module in model.named_modules():
    if isinstance(module, cls):
      names = name.split('.')
      lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
      lora_module_names.remove('lm_head')
  return list(lora_module_names)

modules = find_all_linear_names(model)

##Making lora Model
lora_config = LoraConfig(
    r=8,                                   # Number of quantization levels
    lora_alpha=32,                         # Hyperparameter for LoRA
    target_modules = modules, # Modules to apply LoRA to
    lora_dropout=0.05,                     # Dropout probability
    bias="none",                           # Type of bias
    task_type="CAUSAL_LM"                  # Task type (in this case, Causal Language Modeling)
)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)


print("Number of the trained weights.............................")
trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")

tokenizer.pad_token = tokenizer.eos_token
torch.cuda.empty_cache()

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=test_data,
    dataset_text_field="prompt",
    peft_config=lora_config,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        fp16=True,
        warmup_steps=0.03,
        max_steps=2000,
        learning_rate=2e-4,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        save_strategy="epoch",
        evaluation_strategy = "epoch",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
print("finetuning starts....................")
model.config.use_cache = False # set to False as we're going to use gradient checkpointing, set to true for inferance
trainer.train()

print("uploading to hub...................")
model.push_to_hub(adapter_hub_name)
tokenizer.push_to_hub(adapter_hub_name)


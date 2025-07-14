#!/usr/bin/env python
# coding: utf-8

# # L3: Supervised Fine-Tuning (SFT)

# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>(Kernel Starting)</code>:</b> This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.</p>

# In[ ]:


# Warning control
import warnings
warnings.filterwarnings('ignore')

# Import libraries
import torch
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

# Import helper functions
from utils import generate_responses, test_model_with_questions, load_model_and_tokenizer, display_dataset


# <div style="background-color:#fff6ff; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px">
# <p> 💻 &nbsp; <b>Access <code>requirements.txt</code> file:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Open"</em>.</p>
# 
# <p> ⬇ &nbsp; <b>Download Notebooks:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Download as"</em> and select <em>"Notebook (.ipynb)"</em>.</p>
# 
# <p> 📒 &nbsp; For more help, please see the <em>"Appendix – Tips, Help, and Download"</em> Lesson.</p>
# </div>

# Helper functions are now imported from utils.py


# ## Load base model & test on simple questions

# In[ ]:


USE_GPU = False

questions = [
    "Give me an 1-sentence introduction of LLM.",
    "Calculate 1+1-1",
    "What's the difference between thread and process?"
]


# In[ ]:


model, tokenizer = load_model_and_tokenizer("Qwen/Qwen3-0.6B-Base", USE_GPU)

test_model_with_questions(model, tokenizer, questions, 
                          title="Base Model (Before SFT) Output")

del model, tokenizer


# ## SFT results on Qwen3-0.6B model
# 
# In this section, we're reviewing the results of a previously completed SFT training. Due to limited resources, we won’t be running the full training on a relatively large model like Qwen3-0.6B. However, in the next section of this notebook, you’ll walk through the full training process using a smaller model and a lightweight dataset.

# In[ ]:


model, tokenizer = load_model_and_tokenizer("banghua/Qwen3-0.6B-SFT", USE_GPU)

test_model_with_questions(model, tokenizer, questions, 
                          title="Base Model (After SFT) Output")

del model, tokenizer


# ## Doing SFT on a small model

# <div style="background-color:#fff6ff; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px">
# <p> 💻 &nbsp; <b>Note:</b> We're performing SFT on a small model <code>HuggingFaceTB/SmolLM2-135M</code> and a smaller training dataset to to ensure the full training process can run on limited computational resources. If you're running the notebooks on your own machine and have access to a GPU, feel free to switch to a larger model—such as <code>Qwen/Qwen3-0.6B-Base</code>—to perform full SFT and reproduce the results shown above.</p>
# </div>

# In[ ]:


model_name = "HuggingFaceTB/SmolLM2-135M"
model, tokenizer = load_model_and_tokenizer(model_name, USE_GPU)


# In[ ]:


train_dataset = load_dataset("banghua/DL-SFT-Dataset")["train"]
if not USE_GPU:
    train_dataset=train_dataset.select(range(100))

display_dataset(train_dataset)


# In[ ]:


# SFTTrainer config 
sft_config = SFTConfig(
    learning_rate=8e-5, # Learning rate for training. 
    num_train_epochs=1, #  Set the number of epochs to train the model.
    per_device_train_batch_size=1, # Batch size for each device (e.g., GPU) during training. 
    gradient_accumulation_steps=8, # Number of steps before performing a backward/update pass to accumulate gradients.
    gradient_checkpointing=False, # Enable gradient checkpointing to reduce memory usage during training at the cost of slower training speed.
    logging_steps=2,  # Frequency of logging training progress (log every 2 steps).
    bf16=False,  # Disable bf16 for CPU training
    fp16=False,  # Disable fp16 for CPU training
)


# In[ ]:


sft_trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset, 
    processing_class=tokenizer,
)
sft_trainer.train()


# ## Testing training results on small model and small dataset
# 
# **Note:** The following results are for the small model and dataset we used for SFT training, due to limited computational resources. To view the results of full-scale training on a larger model, see the **"SFT Results on Qwen3-0.6B Model"** section above.

# In[ ]:


if not USE_GPU: # move model to CPU when GPU isn’t requested
    sft_trainer.model.to("cpu")
test_model_with_questions(sft_trainer.model, tokenizer, questions, 
                          title="Base Model (After SFT) Output")





import re
import time
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from tokenizers import Tokenizer
from colorama import Fore

from GPT.model import GPTModel
from BERT.model import UncasedClassifier
from BART.model import BARTModel


def formatted_time(start_time):
    """Formats processing time."""
    elapsed_time = time.time() - start_time
    # Calculate hours, minutes and seconds
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    return f"{Fore.CYAN}{hours}{Fore.RESET} hrs {Fore.CYAN}{minutes}{Fore.RESET} mins {Fore.CYAN}{seconds:.3f}{Fore.RESET} secs"

######################## Global Settings #########################

# Global manual seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)  
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

########################### Model Setup ###########################
        
@dataclass    
class GPTConfig:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16  # bfloat16 or float16
    ntoken = 50304  # Size of tokenizer vocabulary 
    seqlen = 1024  # Length of token sequence per mini-batch
    nbatch = 8  # Number of mini-batches (token sequences)
    dmodel = 768  # Embedding dimension/features
    nhead = 12  # Number of attention heads 
    nlayer = 12  # Number of encoder/decoder layers
    shead = dmodel // nhead # Individual head size
    dropout = 0.1  # % of nodes turned off during training
    bias = False  # Optionally turn bias on/off in linear/normalization layers
    flash = False  # Enable flash attention - need torch.compile and triton package
    load_path = "Language/saved_models/gpt-ultra.pt"
    token_path = "Language/saved_models/bpe-tokenizer.json"
    # Special token ids
    bos_id = 1
    eos_id = 2
    cls_id = 3
    sep_id = 4

gpt_config = GPTConfig() 


@dataclass    
class BERTConfig:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16  # bfloat16 or float16 for autocast
    ntoken = 50304  # Size of tokenizer vocabulary 
    seqlen = 512  # Length of token sequence per mini-batch
    nbatch = 20  # Number of mini-batches (token sequences)
    dmodel = 768  # Embedding dimension/features
    nhead = 12  # Number of attention heads 
    nlayer = 12  # Number of encoder/decoder layers
    shead = dmodel // nhead # Individual head size
    dropout = 0.1  # % of nodes turned off during training
    bias = True  # Optionally turn bias on/off in linear/normalization layers
    nlabels = 29  # Number of NER labels
    load_path = "Language/saved_models/bert-classifier.pt"
    token_path = "Language/saved_models/wp-tokenizer.json"
    # Special token ids
    cls_id = 1
    sep_id = 2

bert_config = BERTConfig()


@dataclass    
class BARTConfig:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16  # bfloat16 or float16
    ntoken = 50304  # Size of tokenizer vocabulary 
    seqlen = 1024  # Length of token sequence per mini-batch (context window)
    nbatch = 8  # Number of mini-batches (token sequences)
    dmodel = 768  # Embedding dimension
    nhead = 6  # Number of attention heads 
    nlayer = 6  # Number of encoder/decoder layers
    shead = dmodel // nhead # Individual head size
    dropout = 0.1  # % of nodes turned off during training
    bias = True  # Optionally turn bias on/off in linear/normalization layers
    # load_path = "Language/saved_models/bart-summarizer.pt"
    load_path = "Language/saved_models/bart-nmt-spanish.pt"
    token_path = "Language/saved_models/bpe-tokenizer.json"
    # Special token ids
    bos_id = 1
    eos_id = 2

bart_config = BARTConfig() 

######################## Inference ##########################

######## Example Using BERT UncassedClassifier Model ########

classifier = UncasedClassifier(bert_config)

text = """his name is tom billig and here are some facts:

1. he was born on july 5th, 1977 in minnapolis, mn.

2. tom served in the us army as an airborne ranger in the 75th ranger regiment.

3. mr. billig graduated from the university of tampa.

at one time, tom wanted to join the cia, but life had different plans in store for him!
"""

output_text, sentiment, entities = classifier(text, color=True)

print(f'{Fore.GREEN}Input Text:{Fore.RESET} {text}')
print(f'{Fore.GREEN}Output Text:{Fore.RESET} {output_text}')
print(f'{Fore.GREEN}Tagged Entities:{Fore.RESET} {entities}')
print(f"{Fore.GREEN}Negative:{Fore.RESET} {sentiment['negative']:.3f}",
    f"{Fore.GREEN}Neutral:{Fore.RESET} {sentiment['neutral']:.3f}",
    f"{Fore.GREEN}Positive:{Fore.RESET} {sentiment['positive']:.3f}")


######### Example Using BART Model for Translation ##########

bart_model = BARTModel.from_pretrained(bart_config)
tokenizer = Tokenizer.from_file(bart_config.token_path)

text = "This [MASK] simple inference [MASK] using [MASK] tokens."
print(f'{Fore.GREEN}Inference Input:{Fore.RESET} {text}')
encoded = tokenizer.encode(text).ids
input_ids = [1] + encoded + [2]

infer_ids = bart_model.inference(input_ids)
infer_text = tokenizer.decode(infer_ids)
print(f'{Fore.GREEN}Inference Output:{Fore.RESET} {infer_text.strip()}')

prompt = "Hola, como estas? Me llamo Tomas. Como te llamas?"
print(f'{Fore.GREEN}Generation Prompt:{Fore.RESET} {prompt}')
encoded = tokenizer.encode(prompt).ids
prompt_ids = [1] + encoded + [2]

generated_ids = bart_model.generate(prompt_ids, temp=1.0, top_k=None)
generated_text = tokenizer.decode(generated_ids) 

print(f'{Fore.GREEN}Generation Output:{Fore.RESET} {generated_text.strip()}')


####### Example Using GPT and BERT Classifier Models ########

gpt_model = GPTModel.from_pretrained(gpt_config)
tokenizer = Tokenizer.from_file(gpt_config.token_path)

chat = True
while chat:

    choice = input(f'{Fore.YELLOW}Do you want to chat (y or n)? {Fore.RESET}')
    if choice == 'y':
        text = input(f'{Fore.GREEN}Prompt:{Fore.RESET} ')

        start_time = time.time()

        encoded = tokenizer.encode(text).ids
        prompt = [1] + encoded + [4] # Insert [BOS] and [SEP] special tokens

        generated_ids = gpt_model.generate(prompt, temp=1.0, top_k=None, top_p=None)
        generated_text = tokenizer.decode(generated_ids[len(prompt):])

        output_text, _, _ = classifier(generated_text, color=True)

        print(f'{Fore.GREEN}Response:{Fore.RESET} {generated_text}')

        print(f'Process Time: {time.time() - start_time}')
    else:
        chat = False
        break
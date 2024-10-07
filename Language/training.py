import os
import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass

from GPT.trainer import GPTTrainer
from BERT.trainer import BERTPreTrainer, BERTFineTuner
from BART.trainer import BARTTrainer

########################## Global Settings ###########################

plt.style.use('dark_background')

# Global manual seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)  
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

# Set distributed training environment variables 
os.environ['MASTER_ADDR'] = '192.168.0.182'  # Master node IP
os.environ['MASTER_PORT'] = '3175'  # Same port for all processes
os.environ['WORLD_SIZE'] = '2'  # Total number of GPUs across all machines
os.environ['RANK'] = '0'  # Change this to 0 for master, 1 for worker on the other machine

########################### GPT Model Setup ###########################
        
@dataclass    
class GPTConfig:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16  # bfloat16 or float16
    ntoken = 50304  # Size of tokenizer vocabulary 
    seqlen = 1280  # Length of token sequence per mini-batch
    nbatch = 8  # Number of mini-batches (token sequences)
    dmodel = 1024  # Embedding dimension
    nhead = 16  # Number of attention heads 
    nlayer = 12  # Number of encoder/decoder layers
    shead = dmodel // nhead # Individual head size
    dropout = 0.1  # % of nodes turned off during training
    bias = False  # Optionally turn bias on/off in linear/normalization layers
    flash = False  # Enable flash attention - only in WSL / Linux builds
    data_dir = "C:\\Users\\Tom\\Datasets\\pre-train\\fineweb"
    load_path = "Language/saved_models/gpt-pretrain-2.pt"
    save_path = "Language/saved_models/gpt-pretrain-2.pt"
    token_path = "Language/saved_models/bpe-tokenizer.json"
    lr_sched = 'decay' # warmup, cyclic, decay, rlrp, or none
    grad_accum = 8  # Gradient accumulation steps to simulate larger batch sizes
    max_iters = 1000  # Max training iterations
    wu_iters = 100 // grad_accum  # Warmup iterations for warmup scheduler
    sched_iters = 1000 // grad_accum  # Adjust scheduler iters for gradient accumulation
    max_lr = 6e-4  # Max or starting learning rate
    min_lr = 5e-4  # Minimum learning rate
    step_up = 1000 // grad_accum  # Step up iterations for CLR scheduler
    step_dn = 2000  // grad_accum  # Step down iterations
    epochs = 1  # Default to one training loop through the train dataset
    # Special tokens
    bos_id = 1
    eos_id = 2
    cls_id = 3
    sep_id = 4

########################### BERT Model Setup ###########################
        
@dataclass    
class BERTConfig:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16  # bfloat16 or float16 for autocast
    ntoken = 50304  # Size of tokenizer vocabulary 
    seqlen = 512  # Length of token sequence per mini-batch
    nbatch = 20  # Number of mini-batches (token sequences)
    dmodel = 768  # Embedding dimension
    nhead = 12  # Number of attention heads 
    nlayer = 12  # Number of encoder/decoder layers
    shead = dmodel // nhead # Individual head size
    dropout = 0.1  # % of nodes turned off during training
    bias = True  # Optionally turn bias on/off in linear/normalization layers
    nlabels = 29  # Number of NER labels
    # data_dir = "D:\\Datasets\\pre-train\\bert" # Pretraining
    data_dir = "C:\\Users\\Tom\\Desktop\\VS Code Projects\\Python Projects\\DeepLearning\\data\\ner-sent-dataset.csv"
    load_path = "Language/saved_models/bert-pretrain.pt"
    save_path = "Language/saved_models/bert-classifier.pt"
    token_path = "Language/saved_models/wp-tokenizer.json"
    lr_sched = 'warmup' # warmup, cyclic, decay, rlrp, or none
    grad_accum = 4  # Gradient accumulation steps to simulate larger batch sizes (4 x 20)
    max_iters = 14000  # Max training iterations
    wu_iters = 1000 // grad_accum  # Warmup iterations for warmup scheduler
    sched_iters = 14000 // grad_accum  # Adjust scheduler iters for gradient accumulation 
    base_lr = 9e-6 # Base model param groups 0 and 1 learning rate
    max_lr = 9e-6  # Max or starting learning rate
    min_lr = 1e-6  # Minimum learning rate
    step_up = 2000 // grad_accum # Step up iterations for CLR scheduler
    step_dn = 3000 // grad_accum # Step down iterations
    epochs = 4  # Default to one training loop through the train dataset


########################### BART Model Setup ###########################
        
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
    # data_dir = "C:\\Users\\Tom\\Datasets\\pre-train\\fineweb"
    # data_dir = "D:\\Datasets\\fine-tune\\translation\\spanish"
    data_dir = "D:\\Datasets\\fine-tune\\summarization"
    load_path = "Language/saved_models/bart-pretrain.pt"
    save_path = "Language/saved_models/bart-summarizer.pt"
    token_path = "Language/saved_models/bpe-tokenizer.json"
    lr_sched = 'warmup' # warmup, cyclic, decay, rlrp or none
    grad_accum = 5  # Gradient accumulation steps to simulate larger batch sizes (5 x 9)
    max_iters = 100000  # Max training iterations
    wu_iters = 1000 // grad_accum  # Warmup iterations for warmup scheduler
    sched_iters = 100000 // grad_accum  # Adjust scheduler iters for gradient accumulation
    max_lr = 9e-6  # Max or starting learning rate
    min_lr = 1e-6  # Minimum learning rate
    step_up = 2000 // grad_accum  # Step up iterations for CLR scheduler
    step_dn = 3000  // grad_accum  # Step down iterations
    epochs = 1  # Default to one training loop through the train dataset
    # BART De-noising Pre-training Tasks
    token_mask = True # Randomly mask single and multiple token spans
    sent_perm = True # Shuffle sentence order in the document
    token_perm = False # Shuffle token order in each sentence (advanced)
    # Special tokens
    bos_id = 1
    eos_id = 2


if __name__ == '__main__':

    #### Code to check required backends for torch distributed training ####
    # import torch
    # import torch.distributed as dist
    # print(torch.__version__)
    # print(torch.cuda.is_available())
    # print(torch.backends.cudnn.is_available())
    # print(dist.is_gloo_available())
    # print(dist.is_nccl_available())

    #### Example usage with 4 lines of code to pre-train a GPT model from scratch ####
    # config = GPTConfig() 
    # trainer = GPTTrainer(config, state=None, pretrain=True, distributed=False)
    # trainer.train(save_cp=True, best_model=False, plot_results=True)
    # trainer.test()

    ##### Example with BERT #####
    # config = BERTConfig() 
    # trainer = BERTPreTrainer(config, state=None, distributed=False)
    # trainer = BERTFineTuner(config, state=None, distributed=False)
    # trainer.train(save_cp=True, best_model=True, plot_results=True)
    # trainer.test()

    ##### Example with BART #####
    config = BARTConfig() 
    trainer = BARTTrainer(config, state='weights', pretrain=False, single_file=False, distributed=False)
    trainer.train(save_cp=True, best_model=False, plot_results=True)
    text = "WASHINGTON (CNN)  -- A leader of the conservative ""Blue Dog"" Democrats told CNN Wednesday he and other group members may vote to block House Democrats' health care bill from passing a key committee if they don't get some of the changes they want. Rep. Mike Ross, D-Arkansas, is a leading negotiator for the Blue Dog Democrats on health care. ""We remain opposed to the current bill, and we continue to meet several times a day to decide how we're going to proceed and what amendments we will be offering as Blue Dogs on the committees,"" said Rep. Mike Ross, D-Arkansas. Ross said the bill unveiled Tuesday by House Democratic leaders did not address concerns he and other conservative Democrats outlined in a letter late last week to Speaker Nancy Pelosi. The conservative Democrats don't believe the legislation contains sufficient reforms to control costs in the health care system and believe additional savings can be found. Their letter to leaders raised concerns about new mandates on small businesses. Blue Dogs also say the bill fails to fix the inequities in the current system for health care costs for rural doctors and hospitals. The Energy and Commerce committee, along with two other House committees, is scheduled to take up the bill Thursday. Democrats outnumber Republicans 36-23 on the Energy and Commerce committee, which contains eight Blue Dogs, including Ross. If seven Democrats vote with Republicans against the bill, it would fail to advance to the House floor. Asked whether the Blue Dogs on Energy and Commerce are considering voting as a group against the bill if it remains unchanged, Ross replied, ""absolutely."" He didn't give details on changes the Blue Dogs want. But he did say he wasn't satisfied with the penalty exemption for small businesses that don't provide health insurance for employees. An earlier draft of the Democrats' bill exempted businesses from paying a penalty if their payrolls were less than $100,000. Democratic leaders raised that payroll amount to $250,000."
    trainer.test(prompt=text)
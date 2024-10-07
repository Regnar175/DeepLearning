import os
import torch
from colorama import Fore
from tokenizers import Tokenizer
import pandas as pd

from utilities import Loader


class GPTPreTrainLoader(Loader):
    """
    Custom Dataset/Dataloader class for pre-training on large text file datasets.
    Model parameters based on a single GPU (RTX4090 or RTX3090) with 24GB of memory.\n
    Params:
        `config` = model configuration data class (see attribute notes below)
    Attributes:
        device: str = 'cpu' or 'cuda' | used to set the device.
        tokenizer: object = Hugging Face custom trained tokenizer loaded from json file.
        vocab_size: int = 50,304 | size of the pre-trained tokenizer vocabulary.
        bos_id: int = 1 | [BOS] special token ID
        eos_id: int = 2 | [EOS] special token ID
        batch_size: int = 8 | max batch size for GPT w/ 768 dim, 12 heads, & 12 layers.
        seq_len: int = 1,024 | max sequence length for GPT w/ 768 dim, 12 heads, & 12 layers.
        batch_len: int = (batch_size * seq_len) - batch_size | length used to prepare batches.
        epochs: int = Any | set the number of training epochs (for smaller datasets).
        epoch_count: int = 0 | epoch counter used to track number of training epochs.
        train_files: list = Any | list of file paths used to load the next train text file.
        valid_files: list = Any | list of file paths used to load the next valid text file.
        train_data: list = Any | list of prepared train batches by batch_len (batch_size * seq_len).
        valid_data: list = Any | list of prepared valid batches by batch_len (batch_size * seq_len).
        train_file_idx: int = 0 | current train file index used for tracking and loading new data files.
        train_data_idx: int = 0 | current train data list index used for loading the next batch.
        valid_file_idx: int = 0 | current valid file index used for tracking and loading new data files.
        valid_data_idx: int = 0 | current valid data list index used for loading the next batch.
        num_replicas: int = 1 | number of GPUs for distributed training.
    """
    def __init__(self, config):
        super().__init__(config)
        self.bos_id = self.tokenizer.token_to_id('[BOS]')
        self.eos_id = self.tokenizer.token_to_id('[EOS]')
        # Load the inital train/valid data lists
        self.load_train_data()
        self.load_valid_data()


    def prepare_data(self, file_path):
        """Prepare a data file and create a list of batches to call from."""
        # Read in the next text file and prepare the data into a list
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            token_ids = self.tokenizer.encode(text).ids

        # Ensure that the batch is evenly divisible by the number of GPUs
        if self.batch_size % self.num_replicas != 0:
            raise ValueError(f"Batch size [{self.batch_size}] is not evenly divisible by the number of replicas [{self.num_replicas}]")
        
        dataset = []
        # Chunk batches by total batch length minus 1 token per sequence
        for idx in range(0, len(token_ids), self.batch_len):
            batch = token_ids[idx:idx + self.batch_len]
            if len(batch) == self.batch_len:
                dataset.append(batch)
        
        return dataset
    
    def prepare_batch(self, batch):
        """Prepare input_ids and target batches."""

        batch = torch.tensor(batch, dtype=torch.long) 
        
        input_ids = batch.view(self.batch_size, self.seq_len - 1)
        target = input_ids.clone()

        # Add [BOS] token at the start of the sequence for input_ids batch
        bos = torch.full((input_ids.size(0), 1), self.bos_id, dtype=torch.long) 
        input_ids = torch.cat((bos, input_ids), dim=-1)
        # Add [EOS] token at the end of the sequence for target batch
        eos = torch.full((target.size(0), 1), self.eos_id, dtype=torch.long) 
        target = torch.cat((target, eos), dim=-1)
    
        if self.device == 'cuda':
            input_ids = input_ids.pin_memory().to(self.device, non_blocking=True)
            target = target.pin_memory().to(self.device, non_blocking=True)

        return {'input_ids': input_ids, 'target': target}


class GPTFineTuneLoader(Loader):
    """
    Custom Dataset/Dataloader class for fine-tuning on large csv file datasets.
    Model parameters based on a single GPU (RTX4090 or RTX3090) with 24GB of memory.\n
    Params:
        `config` = model configuration data class (see attribute notes below)
    Attributes:
        device: str = 'cpu' or 'cuda' | used to set the device.
        tokenizer: object = Hugging Face custom trained tokenizer loaded from json file.
        vocab_size: int = 50,304 | size of the pre-trained tokenizer vocabulary.
        bos_id: int = 1 | [BOS] special token ID
        eos_id: int = 2 | [EOS] special token ID
        batch_size: int = 8 | max batch size for GPT w/ 768 dim, 12 heads, & 12 layers.
        seq_len: int = 1,024 | max sequence length for GPT w/ 768 dim, 12 heads, & 12 layers.
        batch_len: int = (batch_size * seq_len) - batch_size | length used to prepare batches.
        epochs: int = Any | set the number of training epochs (for smaller datasets).
        epoch_count: int = 0 | epoch counter used to track number of training epochs.
        train_files: list = Any | list of file paths used to load the next train text file.
        valid_files: list = Any | list of file paths used to load the next valid text file.
        train_data: list = Any | list of prepared train batches by batch_len (batch_size * seq_len).
        valid_data: list = Any | list of prepared valid batches by batch_len (batch_size * seq_len).
        train_file_idx: int = 0 | current train file index used for tracking and loading new data files.
        train_data_idx: int = 0 | current train data list index used for loading the next batch.
        valid_file_idx: int = 0 | current valid file index used for tracking and loading new data files.
        valid_data_idx: int = 0 | current valid data list index used for loading the next batch.
        num_replicas: int = 1 | number of GPUs for distributed training.
    """
    def __init__(self, config):
        super().__init__(config)
        self.bos_id = self.tokenizer.token_to_id('[BOS]')
        self.eos_id = self.tokenizer.token_to_id('[EOS]')
        self.sep_id = self.tokenizer.token_to_id('[SEP]')
        # Load the inital train/valid data lists
        self.load_train_data()
        self.load_valid_data()


    def prepare_data(self, file_path):
        """Prepare a data file and create a list of batches to call from."""
        # Read in the next csv file and prepare the data into a list of batches

        self.tokenizer.enable_padding(pad_id=0, length=self.config.seqlen)
        self.tokenizer.enable_truncation(max_length=self.config.seqlen)

        df = pd.read_csv(file_path, header=None)
        samples = df[0][1:].to_list()

        sequences = self.tokenizer.encode_batch(samples)

        # Ensure that the batch is evenly divisible by the number of GPUs
        if self.batch_size % self.num_replicas != 0:
            raise ValueError(f"Batch size [{self.batch_size}] is not evenly divisible by the number of replicas [{self.num_replicas}]")

        dataset = []
        for idx in range(0, len(sequences), self.batch_size):
            batch = sequences[idx:idx + self.batch_size]
            if len(batch) == self.batch_size:
                dataset.append(batch)
        
        return dataset
    
    def prepare_batch(self, batch):
        """Prepare input_ids, attn_mask, and target batches."""

        input_ids = [sequence.ids for sequence in batch]
        attn_mask = [sequence.attention_mask for sequence in batch]

        input_ids = torch.tensor(input_ids, dtype=torch.long) 
        attn_mask = torch.tensor(attn_mask, dtype=torch.long) 
            
        target = input_ids[:, 1:].clone() # Shift the sequence left 1 token
        # Create a column of pad tokens to insert at end of target batch
        pad_column = torch.full((target.size(0), 1), 0, dtype=torch.long) 
        target = torch.cat((target, pad_column), dim=-1)
    
        if self.device == 'cuda':
            input_ids = input_ids.pin_memory().to(self.device, non_blocking=True)
            target = target.pin_memory().to(self.device, non_blocking=True)
            attn_mask = attn_mask.pin_memory().to(self.device, non_blocking=True)

        return {'input_ids': input_ids, 'target': target, 'attn_mask': attn_mask}
    



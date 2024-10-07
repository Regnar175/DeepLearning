import re
import os
import random
import torch
from colorama import Fore
from tokenizers import Tokenizer
from nltk.tokenize import sent_tokenize
import pandas as pd
from tqdm import tqdm

from utilities import Loader


class BARTPreTrainLoader(Loader):
    """
    Custom Dataset/Dataloader class for pre-training on large text file datasets.
    Model parameters based on a single GPU (RTX4090 or RTX3090) with 24GB of memory.\n
    Params:
        `config` = model configuration data class (see attribute notes below)
    Attributes:
        tokenizer: object = Hugging Face custom trained tokenizer loaded from json file.
        vocab_size: int = 50,304 | size of the pre-trained tokenizer vocabulary.
        pad_id: int = 0 | [PAD] special token ID
        bos_id: int = 1 | [BOS] special token ID
        eos_id: int = 2 | [EOS] special token ID
        mask_id: int = 2 | [MASK] special token ID
        batch_size: int = 8 | max batch size for BART w/ 768 dim, 6 heads, & 6 layers.
        seq_len: int = 1,024 | max sequence length for BART w/ 768 dim, 6 heads, & 6 layers.
        device: str = 'cpu' or 'cuda' | used to set the device.
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
        self.pad_id = self.tokenizer.token_to_id('[PAD]')
        self.bos_id = self.tokenizer.token_to_id('[BOS]')
        self.eos_id = self.tokenizer.token_to_id('[EOS]')
        self.mask_id = self.tokenizer.token_to_id('[MASK]')
        # Pre-training tasks
        self.token_mask = config.token_mask
        self.sent_perm = config.sent_perm
        self.token_perm = config.token_perm
        # Load the inital train/valid data lists
        self.load_train_data()
        self.load_valid_data()
        
    
    def prepare_data(self, file_path):
        """Prepare a data file and create a list of batches to call from."""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            # Split file into documents by targeting the [EOD] token
            documents = re.split(r'\[EOD\]\n', text)

        # Ensure that the batch is evenly divisible by the number of GPUs
        if self.batch_size % self.num_replicas != 0:
            raise ValueError(f"Batch size [{self.batch_size}] is not evenly divisible by the number of replicas [{self.num_replicas}]")
        
        dataset = []
        batch = []
        sequence = []
        seq_count = 0

        for doc in documents:
            sentences = sent_tokenize(doc)
            # Encode each sentence seperately for downstream batch preparation
            for sent in sentences:
                encoded = self.tokenizer.encode(sent).ids
                seq_count += len(encoded)
                sequence.append(encoded)

            if sequence: 
                # Account for [BOS] & [EOS] tokens
                if seq_count <= (self.seq_len - 2):
                    batch.append(sequence)
                # Reset the sequence and counter variables
                sequence = []
                seq_count = 0
            # Add the batch to the dataset
            if len(batch) == self.batch_size:
                dataset.append(batch)
                batch = []

        return dataset

    def mask_tokens(self, sentence, mask_prob=0.30):
        """Mask random tokens/phrases in the sentence with a [MASK] token 
        excluding special tokens and punctuation."""

        # Use an infill flag to skip adding the next token after a masked token
        infill = False
        masked_sent = []
        for token in sentence:
            # Avoid masking special tokens (ids 0 to 7)
            if token > 7: 
                if random.random() < 0.05:
                    continue # Randomly delete a token 5% of the time
                if infill: # Flag to skip adding the next token after [MASK]
                    infill = False # Reset the flag
                    continue
                if random.random() < mask_prob:
                    masked_sent.append(self.mask_id)
                    # 50% chance to skip adding the next token after masked
                    if random.random() > 0.5:
                        infill = True # Set the flag to skip next token
                else:
                    masked_sent.append(token)
            else:
                masked_sent.append(token)

        return masked_sent

    def prepare_batch(self, batch):
        """Prepare corrupted inputs, attn_masks, and target batches."""
        input_batch = []
        target_batch = []
        input_mask_batch = []
        target_mask_batch = []

        for sequence in batch:
            masked_seq = sequence.copy()
            # Shuffle sentences for sentence permutation task
            if self.sent_perm:
                random.shuffle(masked_seq) 
            input_seq = []
            target_seq = []
            
            for sent in sequence:
                # Keep original uncorrupted token sequence for target batch
                target_seq.extend(sent)

            for sent in masked_seq:
                # Mask/infill tokens in each sentence
                if self.token_mask:
                    sent = self.mask_tokens(sent)
                # Shuffle tokens in each sentence
                if self.token_perm:
                    random.shuffle(sent)
                input_seq.extend(sent)
            
            # Add the [BOS] & [EOS] token and pad sequences to the required sequence length
            input_seq.insert(0, self.bos_id)
            input_seq.extend([self.eos_id])
            # Prepare input mask
            input_mask_seq = [1] * len(input_seq)

            # Pad input sequence
            input_pad = self.seq_len - len(input_seq)
            input_seq.extend([self.pad_id] * input_pad)

            # Add special tokens to the target sequence
            target_seq.insert(0, self.bos_id)
            target_seq.extend([self.eos_id])
            # Prepare target mask
            target_mask_seq = [1] * len(target_seq)

            # Pad target sequence
            target_pad = self.seq_len - len(target_seq) 
            target_seq.extend([self.pad_id] * target_pad)

            # Pad attention masks
            input_mask_seq.extend([self.pad_id] * input_pad)
            target_mask_seq.extend([self.pad_id] * target_pad)
            
            # Append padded sequence lengths to each batch
            input_batch.append(input_seq)
            target_batch.append(target_seq)
            input_mask_batch.append(input_mask_seq)
            target_mask_batch.append(target_mask_seq)

        # Convert each batch to tensors
        input_batch = torch.tensor(input_batch, dtype=torch.long)
        target_batch = torch.tensor(target_batch, dtype=torch.long)
        input_mask_batch = torch.tensor(input_mask_batch, dtype=torch.long)
        target_mask_batch = torch.tensor(target_mask_batch, dtype=torch.long)
        
        if self.device == 'cuda':
            input_ids = input_batch.pin_memory().to(self.device, non_blocking=True)
            target = target_batch.pin_memory().to(self.device, non_blocking=True)
            input_mask = input_mask_batch.pin_memory().to(self.device, non_blocking=True)
            target_mask = target_mask_batch.pin_memory().to(self.device, non_blocking=True)
        
        return {'input': input_ids, 
                'target': target,
                'mask': (input_mask, target_mask)}


class BARTFineTuneLoader(Loader):
    """
    Custom Dataset/Dataloader class for pre-training on large text file datasets.
    Model parameters based on a single GPU (RTX4090 or RTX3090) with 24GB of memory.\n
    Params:
        `config` = model configuration data class (see attribute notes below)
        `single_file` = False | Single file for fine-tuning tasks such as GEC or language datasets
    Attributes:
        tokenizer: object = Hugging Face custom trained tokenizer loaded from json file.
        vocab_size: int = 50,304 | size of the pre-trained tokenizer vocabulary.
        pad_id: int = 0 | [PAD] special token ID
        bos_id: int = 1 | [BOS] special token ID
        eos_id: int = 2 | [EOS] special token ID
        batch_size: int = 8 | max batch size for BART w/ 768 dim, 6 heads, & 6 layers.
        seq_len: int = 1,024 | max sequence length for BART w/ 768 dim, 6 heads, & 6 layers.
        device: str = 'cpu' or 'cuda' | used to set the device.
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
    def __init__(self, config, single_file=False):
        super().__init__(config)
        self.single_file = single_file
        self.pad_id = self.tokenizer.token_to_id('[PAD]')
        self.bos_id = self.tokenizer.token_to_id('[BOS]')
        self.eos_id = self.tokenizer.token_to_id('[EOS]')
        # Load the inital train/valid data lists
        if self.single_file:
            self.prepare_data(config.data_dir)
        else:
            self.load_train_data()
            self.load_valid_data()
        

    def prepare_data(self, file_path):
        """Prepare a data file and create a list of batches to call from."""
        # Read in the next csv file and convert the dataset into a list
        dataset = pd.read_csv(file_path).values.tolist()

        # Ensure that the batch is evenly divisible by the number of GPUs
        if self.batch_size % self.num_replicas != 0:
            raise ValueError(f"Batch size [{self.batch_size}] is not evenly divisible by the number of replicas [{self.num_replicas}]")
        
        processed_data = []
        input_ids, input_mask, target_ids, target_mask = [], [], [], []
        if self.single_file:
            print(f"{Fore.MAGENTA}Loading dataset...{Fore.RESET}")
        for sample in dataset:
            input_seq = self.tokenizer.encode(str(sample[0])).ids
            target_seq = self.tokenizer.encode(str(sample[1])).ids

            if len(input_seq) <= (self.seq_len - 2) and len(target_seq) <= (self.seq_len - 2):
                # Add the [BOS] & [EOS] token and pad sequences to the required sequence length
                input_seq.insert(0, self.bos_id)
                input_seq.extend([self.eos_id])
                # Prepare input mask
                input_mask_seq = [1] * len(input_seq)

                # Pad input sequence
                input_pad = self.seq_len - len(input_seq)
                input_seq.extend([self.pad_id] * input_pad)

                # Add special tokens to the target sequence
                target_seq.insert(0, self.bos_id)
                target_seq.extend([self.eos_id])
                # Prepare target mask
                target_mask_seq = [1] * len(target_seq)

                # Pad target sequence
                target_pad = self.seq_len - len(target_seq) 
                target_seq.extend([self.pad_id] * target_pad)

                # Pad attention masks
                input_mask_seq.extend([self.pad_id] * input_pad)
                target_mask_seq.extend([self.pad_id] * target_pad)

                # Append padded sequence lengths to each batch
                input_ids.append(input_seq)
                target_ids.append(target_seq)
                input_mask.append(input_mask_seq)
                target_mask.append(target_mask_seq)
            else:
                continue

            if len(input_ids) == self.batch_size:
                processed_data.append({'input_ids': input_ids,
                                       'target_ids': target_ids,
                                       'input_mask': input_mask,
                                       'target_mask': target_mask})
                input_ids, input_mask, target_ids, target_mask = [], [], [], []

        if self.single_file:
            # Shuffle and split the data into train and validation sets
            random.shuffle(processed_data)
            split = int(len(processed_data) * 0.9)
            self.train_data = processed_data[:split]
            self.valid_data = processed_data[split:]
        else:
            return processed_data

    def to_tensors(self, batch):
        """Convert batch to tensors."""
        input_ids = torch.tensor(batch['input_ids'], dtype=torch.long)
        target_ids = torch.tensor(batch['target_ids'], dtype=torch.long)
        input_mask = torch.tensor(batch['input_mask'], dtype=torch.long)
        target_mask = torch.tensor(batch['target_mask'], dtype=torch.long)
        
        if self.device == 'cuda':
            input_ids = input_ids.pin_memory().to(self.device, non_blocking=True)
            target_ids = target_ids.pin_memory().to(self.device, non_blocking=True)
            input_mask = input_mask.pin_memory().to(self.device, non_blocking=True)
            target_mask = target_mask.pin_memory().to(self.device, non_blocking=True)
        
        return {'input': input_ids, 
                'target': target_ids,
                'mask': (input_mask, target_mask)}

    def prepare_batch(self, batch):
        """Prepare batches to be called during training and validation steps."""
        return self.to_tensors(batch)
    


class BARTGECLoader:
    """
    Custom Dataset/Dataloader class for fine-tuning on Gramatical Error Correction task.
    Model parameters based on a single GPU (RTX4090 or RTX3090) with 24GB of memory.\n
    Params:
        `config` = model configuration data class (see attribute notes below)
    Attributes:
        tokenizer: object = Hugging Face custom trained tokenizer loaded from json file.
        vocab_size: int = 50,304 | size of the pre-trained tokenizer vocabulary.
        batch_size: int = 8 | max batch size for BART w/ 768 dim, 6 heads, & 6 layers.
        seq_len: int = 1,024 | max sequence length for BART w/ 768 dim, 6 heads, & 6 layers.
        device: str = 'cpu' or 'cuda' | used to set the device.
        epochs: int = Any | set the number of training epochs (for smaller datasets).
        epoch_count: int = 0 | epoch counter used to track number of training epochs.
        train_data: list = Any | list of prepared train mini-batches.
        train_idx: int = 0 | train data list index used for loading the next mini-batch.
        valid_data: list = Any | list of prepared valid mini-batches.
        valid_idx: int = 0 | valid data list index used for loading the next mini-batch.
    """
    def __init__(self, config):
        self.config = config
        self.tokenizer = Tokenizer.from_file(config.token_path)
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.batch_size = config.nbatch
        self.seq_len = config.seqlen
        self.device = config.device
        self.epochs = config.epochs
        self.epoch_count = 0
        self.train_data = []
        self.train_idx = 0
        self.valid_data = []
        self.valid_idx = 0
        # Load the training/validation data
        self.prepare_data(config.data_dir)
        
    def __len__(self):
        # Estimated total number of train batches in the dataset
        return len(self.train_data)

    def __next__(self):
        return self.get_valid_batch()
    
    def prepare_data(self, file_path):
        """Prepare a data file and create a list of batches to call from."""
        print(f"{Fore.MAGENTA}Loading dataset...{Fore.RESET}")
        # Read in the next csv file and convert the dataset into a list
        dataset = pd.read_csv(file_path).values.tolist()

        processed_data = []
        input_ids, input_mask, target_ids, target_mask = [], [], [], []
        for sample in tqdm(dataset, desc='Processing'):

            input_seq = self.tokenizer.encode(str(sample[0])).ids
            target_seq = self.tokenizer.encode(str(sample[1])).ids

            if len(input_seq) <= (self.seq_len - 2): # Account for special tokens
                # Add the [BOS] & [EOS] token and pad sequences to the required sequence length
                input_seq.insert(0, self.tokenizer.token_to_id('[BOS]'))
                input_seq.extend([self.tokenizer.token_to_id('[EOS]')])
                # Prepare input mask
                input_mask_seq = [1] * len(input_seq)

                # Pad input sequence
                input_pad = self.seq_len - len(input_seq)
                input_seq.extend([self.tokenizer.token_to_id('[PAD]')] * input_pad)

                # Add special tokens to the target sequence
                target_seq.insert(0, self.tokenizer.token_to_id('[BOS]'))
                target_seq.extend([self.tokenizer.token_to_id('[EOS]')])
                # Prepare target mask
                target_mask_seq = [1] * len(target_seq)

                # Pad target sequence
                target_pad = self.seq_len - len(target_seq) 
                target_seq.extend([self.tokenizer.token_to_id('[PAD]')] * target_pad)

                # Pad attention masks
                input_mask_seq.extend([0] * input_pad)
                target_mask_seq.extend([0] * target_pad)

                # Append padded sequence lengths to each batch
                input_ids.append(input_seq)
                target_ids.append(target_seq)
                input_mask.append(input_mask_seq)
                target_mask.append(target_mask_seq)
            else:
                continue

            if len(input_ids) == self.batch_size:
                processed_data.append({'input_ids': input_ids,
                                       'target_ids': target_ids,
                                       'input_mask': input_mask,
                                       'target_mask': target_mask})
                input_ids, input_mask, target_ids, target_mask = [], [], [], []

        # Shuffle and split the data into train and validation sets
        random.shuffle(processed_data)
        split = int(len(processed_data) * 0.9)
        self.train_data = processed_data[:split]
        self.valid_data = processed_data[split:]

    def to_tensors(self, batch):
        """Convert batch to tensors"""
        input_ids = torch.tensor(batch['input_ids'], dtype=torch.long)
        target_ids = torch.tensor(batch['target_ids'], dtype=torch.long)
        input_mask = torch.tensor(batch['input_mask'], dtype=torch.long)
        target_mask = torch.tensor(batch['target_mask'], dtype=torch.long)
        
        if self.device == 'cuda':
            input_ids = input_ids.pin_memory().to(self.device, non_blocking=True)
            target_ids = target_ids.pin_memory().to(self.device, non_blocking=True)
            input_mask = input_mask.pin_memory().to(self.device, non_blocking=True)
            target_mask = target_mask.pin_memory().to(self.device, non_blocking=True)
        
        return {'input': input_ids, 
                'target': target_ids,
                'mask': (input_mask, target_mask)}

    def get_train_batch(self):
        """Get the next train batch from the currently loaded train_data list."""
        # Load the next data file if at the end of the current one
        if self.train_idx >= len(self.train_data):
            self.train_idx = 0
            self.epoch_count += 1
            print(f"{Fore.MAGENTA}Epoch: {Fore.RESET}{Fore.CYAN}{self.epoch_count}{Fore.RESET}")

        batch = self.train_data[self.train_idx]

        self.train_idx += 1
        return self.to_tensors(batch)
    
    def get_valid_batch(self):
        """Get the next valid batch from the currently loaded valid_data list."""
        # Load the next data file if at the end of the current one
        if self.valid_idx >= len(self.valid_data):
            self.valid_idx = 0

        batch = self.valid_data[self.valid_idx]

        self.valid_idx += 1
        return self.to_tensors(batch)

    def __iter__(self):
        """Yield a prepared batch per iteration and cycle through all training/validation
        dataset files until the desired epoch has finished or the max number of iterations
        has completed in the training loop."""

        while True:
            if self.epochs is not None and self.epoch_count > self.epochs:
                print(f"{Fore.MAGENTA}Epochs completed: {Fore.RESET}{Fore.CYAN}{self.epoch_count}{Fore.RESET}")
                break  # Stop iterating if the specified number of epochs is reached

            yield self.get_train_batch()

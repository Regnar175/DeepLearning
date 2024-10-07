import ast
import random
import pandas as pd
import torch
from colorama import Fore
from tqdm import tqdm

from utilities import Loader


class BERTPreTrainLoader(Loader):
    """
    Custom Dataset/Dataloader class for pre-training on large text file datasets.
    Model parameters based on a single GPU (RTX4090 or RTX3090) with 24GB of memory.\n
    Params:
        `config` = model configuration data class (see attribute notes below)
    Attributes:
        tokenizer: object = Hugging Face custom trained tokenizer loaded from json file.
        vocab_size: int = 50,304 | size of the pre-trained tokenizer vocabulary.
        batch_size: int = 20 | max batch size for BERT w/ 768 dim, 12 heads, & 12 layers.
        seq_len: int = 512 | max sequence length for BERT w/ 768 dim, 12 heads, & 12 layers.
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
        self.tokenizer.enable_padding(pad_id=0, length=config.seqlen)
        self.tokenizer.enable_truncation(max_length=config.seqlen)
        # Load the inital train/valid data lists
        self.load_train_data()
        self.load_valid_data()

    def generate_arrays(self, token_ids, mask_prob=0.2):
        """Generates input, label, segment and attn_mask arrays from a tokenized/encoded array.\n
        Input: Mask/change 15%-30% of token ids (10% of the sample will not be changed).\n
        Label: Set all ids to -100 that are not prediction token ids\n
        Segment: tag first sentence with 1, second sentence with 2, and [PAD] with 0.\n
        Attn_mask: apply a mask of 1 for all token ids, and 0 for all [PAD] ids.\n
        [PAD] = 0, [CLS] = 1, [SEP] = 2, [MASK] = 3, [UNK] = 4"""
        input_ids, labels, segments, attn_mask = [], [], [], []
        first_sent = True

        for id in token_ids:
            # Handling segments and attention masks
            if id == 0:  # PAD tokens
                input_ids.append(id)
                labels.append(-100)
                segments.append(0)
                attn_mask.append(0)
                continue
            
            # Non-PAD tokens
            segments.append(1 if first_sent else 2)
            attn_mask.append(1)
            
            # Handle [CLS] and [SEP] tokens
            if id in [1, 2]:  # CLS or SEP tokens
                input_ids.append(id)
                labels.append(-100)
                if id == 2 and first_sent:
                    first_sent = False
                continue

            # Skip punctuation for masking
            # Punctuation occupies token ID positions up to 42
            if id <= 42:
                input_ids.append(id)
                labels.append(-100)
                continue
                
            # Masking logic for MLM task
            # Apply masking 15%-30% of the time
            if random.random() < mask_prob:  
                prob = random.random()
                if prob < 0.8:
                    # Replace with [MASK] 80% of the time
                    input_ids.append(3)
                elif prob < 0.9:
                    # Replace with a random token 10% of the time
                    # Avoid using special tokens and punctuation that occupy up to ID 42
                    input_ids.append(random.randint(42, self.vocab_size - 1))
                else:
                    # Keep the original token 10% of the time
                    input_ids.append(id)  
                labels.append(id)
            else:
                input_ids.append(id)
                labels.append(-100)  # Not a token to predict

        return input_ids, labels, segments, attn_mask

    def prepare_data(self, file_path):
        """Prepare a data file and create a list of mini-batches to call from."""
        # Read in the next csv file and prepare the data into a list
        df = pd.read_csv(file_path)
        sent_pairs = list(df.itertuples(index=False, name=None))

        # Ensure that the batch is evenly divisible by the number of GPUs
        if self.batch_size % self.num_replicas != 0:
            raise ValueError(f"Batch size [{self.batch_size}] is not evenly divisible by the number of replicas [{self.num_replicas}]")

        dataset = []
        for pair in sent_pairs:
            s1, s2, is_next = pair
            encoded = self.tokenizer.encode(s1, s2).ids
            mini_batch = self.generate_arrays(encoded)
            mini_batch += (is_next,)
            dataset.append((mini_batch))

        return dataset
    
    def to_tensors(self, batch):
        """Convert batch arrays to tensors."""

        inputs = torch.tensor(batch['input_ids'], dtype=torch.long)
        labels = torch.tensor(batch['labels'], dtype=torch.long)
        segments = torch.tensor(batch['segments'], dtype=torch.long)
        attn_mask = torch.tensor(batch['attn_mask'], dtype=torch.long)
        next_ids = torch.tensor(batch['is_next'], dtype=torch.long)
        
        if self.device == 'cuda':
            inputs = inputs.pin_memory().to(self.device, non_blocking=True)
            labels = labels.pin_memory().to(self.device, non_blocking=True)
            segments = segments.pin_memory().to(self.device, non_blocking=True)
            attn_mask = attn_mask.pin_memory().to(self.device, non_blocking=True)
            next_ids = next_ids.pin_memory().to(self.device, non_blocking=True)

        return {"input_ids": inputs,
                "labels": labels,
                "segments": segments,
                "attn_mask": attn_mask,
                "is_next": next_ids,
                }

    def get_train_batch(self):
        """Prepare and get input, label, segment, attn mask, and is next batches."""
        batch = {
            "input_ids": [],
            "labels": [],
            "segments": [],
            "attn_mask": [],
            "is_next": [],
        }
        for _ in range(self.batch_size):
            # Load the next dataset if at the end of the current one
            if self.train_data_idx >= len(self.train_data):
                self.train_file_idx += 1
                self.train_data_idx = 0
                self.load_train_data()

            mini_batch = self.train_data[self.train_data_idx]
            
            batch['input_ids'].append(mini_batch[0])
            batch['labels'].append(mini_batch[1])
            batch['segments'].append(mini_batch[2])
            batch['attn_mask'].append(mini_batch[3])
            batch['is_next'].append(mini_batch[4])
            # Increment the data index counter
            self.train_data_idx += 1

        return self.to_tensors(batch)

    def get_valid_batch(self):
        """Prepare and get input, label, segment, attn mask, and is next batches."""
        batch = {
            "input_ids": [],
            "labels": [],
            "segments": [],
            "attn_mask": [],
            "is_next": [],
        }
        for _ in range(self.batch_size):
            # Load the next dataset if at the end of the current one
            if self.valid_data_idx >= len(self.valid_data):
                self.valid_file_idx += 1
                self.valid_data_idx = 0
                self.load_valid_data()

            mini_batch = self.valid_data[self.valid_data_idx]
            
            batch['input_ids'].append(mini_batch[0])
            batch['labels'].append(mini_batch[1])
            batch['segments'].append(mini_batch[2])
            batch['attn_mask'].append(mini_batch[3])
            batch['is_next'].append(mini_batch[4])
            # Increment the data index counter
            self.valid_data_idx += 1

        return self.to_tensors(batch)
    
    def __len__(self):
        # Estimated total number of batches in the train dataset
        n_batches = len(self.train_data) / self.batch_size
        est_total_batches = n_batches * len(self.train_files)
        return int(est_total_batches)
    

class BERTClassifierLoader(Loader):
    """
    Custom Dataset/Dataloader class for fine-tuning on NER and sentiment tasks.
    Model parameters based on a single GPU (RTX4090 or RTX3090) with 24GB of memory.\n
    Params:
        `config` = model configuration data class (see attribute notes below)
    Attributes:
        tokenizer: object = Hugging Face custom trained tokenizer loaded from json file.
        vocab_size: int = 50,304 | size of the pre-trained tokenizer vocabulary.
        batch_size: int = 20 | max batch size for BERT w/ 768 dim, 12 heads, & 12 layers.
        seq_len: int = 512 | max sequence length for BERT w/ 768 dim, 12 heads, & 12 layers.
        device: str = 'cpu' or 'cuda' | used to set the device.
        epochs: int = Any | set the number of training epochs (for smaller datasets).
        epoch_count: int = 0 | epoch counter used to track number of training epochs.
        train_data: list = Any | list of prepared train mini-batches.
        valid_data: list = Any | list of prepared valid mini-batches.
        train_idx: int = 0 | train data list index used for loading the next mini-batch.
        valid_idx: int = 0 | valid data list index used for loading the next mini-batch.
    """
    def __init__(self, config):
        super().__init__(config)
        self.train_files = None
        self.valid_files = None
        self.prepare_data(config.data_dir)

    
    def ner_encoder(self, ner_tags, pad_seq=True, seq_len=512):
        encoder = {
            '[PAD]': 0, '[CLS]': 1, '[SEP]': 2, 'O': 3, '##O': 4, 
            'B-PER': 5, '##B-PER': 6, 'I-PER': 7, '##I-PER': 8, 
            'B-LOC': 9, '##B-LOC': 10, 'I-LOC': 11, '##I-LOC': 12, 
            'B-ORG': 13, '##B-ORG': 14, 'I-ORG': 15, '##I-ORG': 16,
            'B-GPE': 17, '##B-GPE': 18, 'I-GPE': 19, '##I-GPE': 20, 
            'B-TIME': 21, '##B-TIME': 22, 'I-TIME': 23, '##I-TIME': 24, 
            'B-MISC': 25, '##B-MISC': 26, 'I-MISC': 27, '##I-MISC': 28,
        }
        ner_tags = ['[CLS]'] + ner_tags + ['[SEP]']
        if pad_seq:
            padding = ['[PAD]'] * (seq_len - len(ner_tags))
            ner_tags += padding
        return [encoder.get(tag, 0) for tag in ner_tags]
    
    def ner_decoder(self, ner_ids, skip_special=True):
        decoder = {
            0: '[PAD]', 1: '[CLS]', 2: '[SEP]', 3: 'O', 4: '##O', 
            5: 'B-PER', 6: '##B-PER', 7: 'I-PER', 8: '##I-PER', 
            9: 'B-LOC', 10: '##B-LOC', 11: 'I-LOC', 12: '##I-LOC', 
            13: 'B-ORG', 14: '##B-ORG', 15: 'I-ORG', 16: '##I-ORG',
            17: 'B-GPE', 18: '##B-GPE', 19: 'I-GPE', 20: '##I-GPE', 
            21: 'B-TIME', 22: '##B-TIME', 23: 'I-TIME', 24: '##I-TIME', 
            25: 'B-MISC', 26: '##B-MISC', 27: 'I-MISC', 28: '##I-MISC', 
        }
        decoded = []
        for id in ner_ids:
            if skip_special:
                if id in [0, 1, 2]:
                    continue
            tag = decoder.get(id, '[UNK]')
            decoded.append(tag)
        return decoded

    def prepare_data(self, file_path):
        """Prepare a data file and create a list of mini-batches to call from."""

        print(f"{Fore.MAGENTA}Loading dataset...{Fore.RESET}")
        # Read in the next csv file and convert the dataset into a list
        dataset = pd.read_csv(file_path).values.tolist()

        # Ensure that the batch is evenly divisible by the number of GPUs
        if self.batch_size % self.num_replicas != 0:
            raise ValueError(f"Batch size [{self.batch_size}] is not evenly divisible by the number of replicas [{self.num_replicas}]")
        
        processed_data = []
        for sample in tqdm(dataset, desc='Processing'):
            tokens = ast.literal_eval(sample[0])
            ner_tags = ast.literal_eval(sample[1])
            sentiment = [float(sample[2]), float(sample[3]), float(sample[4])]

            wp_tags = []
            for word, tag in zip(tokens, ner_tags):
                # Use the tokenizer to split tokens into subwords for alignment
                subwords = self.tokenizer.encode(word, add_special_tokens=False).tokens

                if len(subwords) > 1:
                    wp_tags.append(tag)
                    # Adjust ner tags for subword positions
                    for subword in subwords[1:]:
                        if subword.startswith('##'):
                            wp_tags.append(f'##{tag}')
                        else:
                            if tag.startswith('B-'):
                                tag = tag.replace('B-', 'I-')
                            wp_tags.append(tag)
                else:
                    wp_tags.append(tag)

            if len(wp_tags) <= self.seq_len - 2: # Account for [CLS] and [SEP] tokens
                processed_data.append({'tokens': tokens, 'ner_tags': wp_tags, 'sentiment': sentiment})

        # Shuffle and split the data into train and validation sets
        random.shuffle(processed_data)
        split = int(len(processed_data) * 0.9)
        self.train_data = processed_data[:split]
        self.valid_data = processed_data[split:]

    def to_tensors(self, inpt, mask, ner, sent):
        """Convert batch arrays to tensors."""

        input_ids = torch.tensor(inpt, dtype=torch.long)
        attn_mask = torch.tensor(mask, dtype=torch.long)
        ner_labels = torch.tensor(ner, dtype=torch.long)
        sent_labels = torch.tensor(sent, dtype=torch.float) # Needs to be float for scores
        
        if self.device == 'cuda':
            input_ids = input_ids.pin_memory().to(self.device, non_blocking=True)
            attn_mask = attn_mask.pin_memory().to(self.device, non_blocking=True)
            ner_labels = ner_labels.pin_memory().to(self.device, non_blocking=True)
            sent_labels = sent_labels.pin_memory().to(self.device, non_blocking=True)

        return {'input_ids': input_ids,
                'attn_mask': attn_mask,
                'ner_labels': ner_labels,
                'sent_labels': sent_labels
                }

    def get_train_batch(self):
        """Prepare and return input, mask, ner and sentiment label training batches."""

        self.tokenizer.enable_padding(pad_id=0, length=self.seq_len)

        input_ids, attn_mask, ner_labels, sent_labels, = [], [], [], []
        for _ in range(self.batch_size):
            if self.train_data_idx >= len(self.train_data):
                self.train_data_idx = 0
                self.epoch_count += 1
                print(f"{Fore.MAGENTA}Epoch: {Fore.RESET}{Fore.CYAN}{self.epoch_count}{Fore.RESET}")
            sample = self.train_data[self.train_data_idx]

            encoded = self.tokenizer.encode(sample['tokens'], is_pretokenized=True)
            ner_ids = self.ner_encoder(sample['ner_tags'], seq_len=self.seq_len)
            
            input_ids.append(encoded.ids)
            attn_mask.append(encoded.attention_mask)
            ner_labels.append(ner_ids)
            sent_labels.append(sample['sentiment'])

            self.train_data_idx += 1

        return self.to_tensors(input_ids, attn_mask, ner_labels, sent_labels)

    def get_valid_batch(self):
        """Prepare and return input, mask, ner and sentiment label validation batches."""

        self.tokenizer.enable_padding(pad_id=0, length=self.seq_len)

        input_ids, attn_mask, ner_labels, sent_labels, = [], [], [], []
        for _ in range(self.batch_size):
            if self.valid_data_idx >= len(self.valid_data):
                self.valid_data_idx = 0
            sample = self.valid_data[self.valid_data_idx]

            encoded = self.tokenizer.encode(sample['tokens'], is_pretokenized=True)
            ner_ids = self.ner_encoder(sample['ner_tags'], seq_len=self.seq_len)
            
            input_ids.append(encoded.ids)
            attn_mask.append(encoded.attention_mask)
            ner_labels.append(ner_ids)
            sent_labels.append(sample['sentiment'])

            self.valid_data_idx += 1

        return self.to_tensors(input_ids, attn_mask, ner_labels, sent_labels)
    
    def __len__(self):
        # Estimated total number of batches in the train dataset
        return len(self.train_data) // self.batch_size 






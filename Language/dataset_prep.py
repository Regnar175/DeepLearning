import os
import re
import ast
import time
import json
import random
import pandas as pd
from tokenizers import Tokenizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus.reader import ConllCorpusReader
from colorama import Fore
from datasets import load_dataset
from tqdm import tqdm
import pyarrow.parquet as pq
import shutil

#############################################################################
## This is a collection of useful functions for quickly building datasets. ##
## The functions process the datasets into chunk files for data streaming. ##
## The custom data loader classes are set up to stream in smaller files    ##
## when training on large pre-training datasets such as HF's FineWeb. This ##
## allows for indexing of files to resume pre-training at a later time.    ##
#############################################################################

############################# Utility Functions #############################

def formatted_time(start_time):
    """Formats processing time."""
    elapsed_time = time.time() - start_time
    # Calculate hours, minutes and seconds
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    return f"{Fore.CYAN}{hours}{Fore.RESET} hrs {Fore.CYAN}{minutes}{Fore.RESET} mins {Fore.CYAN}{seconds:.3f}{Fore.RESET} secs"


def get_file_paths(data_dir):
    """Get all files from a file directory and return a list of paths"""
    file_paths = []
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(subdir, file)
            file_paths.append(file_path)
    return file_paths


def get_rand_file_paths(data_dir):
    """Get all files from a file directory and return a shuffled list of paths"""
    file_paths = []
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(subdir, file)
            file_paths.append(file_path)
    random.shuffle(file_paths)
    return file_paths


def concatenate_files(source_dir, dest_dir, prefix, max_files=25, min_size_kb=15):
    """Concatenate dataset files to a desired file size"""

    os.makedirs(dest_dir, exist_ok=True)  # Ensure destination directory exists
    files = [file for file in os.listdir(source_dir) if os.path.getsize(os.path.join(source_dir, file)) >= min_size_kb * 1024]
    concatenated_file_content = ""
    file_count = 0
    for i, file in enumerate(files):
        with open(os.path.join(source_dir, file), 'r', encoding='utf-8') as infile:
            concatenated_file_content += infile.read() + '\n\n'  # Add file content with a separator
        if (i + 1) % max_files == 0 or i == len(files) - 1:  # Every 25 files or the last file
            output_filename = os.path.join(dest_dir, f"{prefix}-{file_count}.txt")
            with open(output_filename, 'w', encoding='utf-8') as outfile:
                outfile.write(concatenated_file_content)
            concatenated_file_content = ""  # Reset for next batch of files
            file_count += 1


def move_rename_files(name):
    """Helper function to move/rename dataset files to a new directory"""
    start_time = time.time()

    train_files = get_file_paths('D:\\Datasets\\pre-train\\train')
    valid_files = get_file_paths('D:\\Datasets\\pre-train\\valid')
    train_dir = f"D:\\Datasets\\pre-train\\{name}\\train"
    valid_dir = f"D:\\Datasets\\pre-train\\{name}\\valid"

    # Train files
    idx = 0
    for file_path in train_files:
        basename = os.path.basename(file_path)
        if name in basename:
            filename = f'{name}-{idx}.txt'
            new_path = os.path.join(train_dir, filename)
            shutil.move(file_path, new_path)
            idx += 1

    # Valid files
    idx = 0
    for file_path in valid_files:
        basename = os.path.basename(file_path)
        if name in basename:
            filename = f'{name}-{idx}.txt'
            new_path = os.path.join(valid_dir, filename)
            shutil.move(file_path, new_path)
            idx += 1

    print(f'Processing time: {formatted_time(start_time)}')


def train_valid_splits(data_dir, shuffle=True):
    """Create train/validation split directories and move files into them"""
    file_paths = get_file_paths(data_dir)
    if shuffle:
        random.shuffle(file_paths)

    split = int(len(file_paths) * 0.9)
    train_files = file_paths[:split]
    valid_files = file_paths[split:]

    for file_path in train_files:
        basename = os.path.basename(file_path)
        new_dir = os.path.join(data_dir, "train")
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        new_file_path = os.path.join(new_dir, basename)
        shutil.move(file_path, new_file_path)

    for file_path in valid_files:
        basename = os.path.basename(file_path)
        new_dir = os.path.join(data_dir, "valid")
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        new_file_path = os.path.join(new_dir, basename)
        shutil.move(file_path, new_file_path)


################################ Pre-Training ###############################

## ============================== Wiki Dataset ==============================

def process_wiki_dataset(output_dir):
    """Download and process the Wiki dataset from Hugging Face"""
    start_time = time.time()
    # Stream the wikipedia dataset from Hugging Face hub
    dataset = load_dataset("wikipedia", "20220301.en", split='train',
                           streaming=True, trust_remote_code=True)

    k = 0  # File counter
    wiki_file = []  # List to hold pre-processed text
    current_size = 0  # Track the size of the accumulated text
    target_size = 3000 * 1024  # Target size in bytes

    print(f'{Fore.GREEN}Beginning Wikipedia Dataset Processing...{Fore.RESET}')
    for sample in tqdm(dataset):
        # Add [EOD] end of document special token to denote end of article
        # Plus some simple pre-processing to format text
        text = re.sub(r'(\n){2,}', '\n\n', text)
        text = re.sub(r'(\s\n){2,}', '\n\n', text)
        text = re.sub(r'\(\)', '', text)
        text = sample['text'] + ' [EOD]' # Add 'End of Document' token at the end of each article
        added_size = len(text.encode('utf-8'))  # Get the size of text in bytes

        # Check if adding this text would exceed the target size
        if current_size + added_size > target_size:
            output_file = os.path.join(output_dir, f"wiki-{k}.txt")
            with open(output_file, 'w', encoding='utf-8') as outfile:
                for doc in wiki_file:
                    outfile.write(doc + '\n\n')
            k += 1
            wiki_file = []  # Reset for the next file
            current_size = 0  # Reset size counter

        wiki_file.append(text)
        current_size += added_size

    # Write any remaining content after the loop
    if wiki_file:
        output_file = os.path.join(output_dir, f"wiki-{k}.txt")
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for doc in wiki_file:
                outfile.write(doc + '\n\n')

    # Create train/valid split directories and move processed files into them
    train_valid_splits(output_dir)

    print(f'{Fore.GREEN}Wikipedia Dataset Processing Complete!{Fore.RESET}',
        f'\nTotal processing time: {formatted_time(start_time)}')


## ============================== Books Dataset ==============================

def process_books_dataset(output_dir):
    """Download and process the Gutenberg (books) dataset from Hugging Face"""
    start_time = time.time()
    # Download the Gutenberg books dataset from Hugging Face and process while streaming
    dataset = load_dataset("BEE-spoke-data/gutenberg-en-v1-clean", split="train", streaming=True)

    k = 0  # File counter
    book_file = []  # List to hold pre-processed text
    current_size = 0  # Track the size of the accumulated text
    target_size = 3000 * 1024  # Target size in bytes

    print(f'{Fore.GREEN}Beginning Books Dataset Processing...{Fore.RESET}')
    for book in tqdm(dataset):
        text = book['text']
        # Simple text pre-processing
        text = re.sub(r'_', '', text)  # Remove all underscores
        text = re.sub(r'(\*\s)+\*', '', text)  # Remove sequences of asterisks
        text = re.sub(r'\[\s*Illustration(?:\:[^\]]*)?\s*\]', '', text)  # Remove illustration placeholders
        text = re.sub(r'(\n){3,}', '\n\n\n', text) # Normalize new lines

        text = text + ' [EOD]' # Add 'End of Document' token after each book
        added_size = len(text.encode('utf-8'))  # Get the size of text in bytes

        # Check if adding this text would exceed the target size
        if current_size + added_size > target_size:
            output_file = os.path.join(output_dir, f"books-{k}.txt")
            with open(output_file, 'w', encoding='utf-8') as outfile:
                for doc in book_file:
                    outfile.write(doc + '\n\n')
            k += 1
            book_file = []  # Reset for the next file
            current_size = 0  # Reset size counter

        # Add the current text after checking for size
        book_file.append(text)
        current_size += added_size

    # Handle any remaining content after the loop
    if book_file:
        output_file = os.path.join(output_dir, f"books-{k}.txt")
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for doc in book_file:
                outfile.write(doc + '\n\n')

    # Create train/valid split directories and move processed files into them
    train_valid_splits(output_dir)

    print(f'{Fore.GREEN}Books Dataset Processing Complete!{Fore.RESET}',
        f'\nTotal processing time: {formatted_time(start_time)}')


## ============================= FineWeb Dataset =============================

def processs_fineweb_dataset(output_dir):
    """Download and process the Fineweb dataset from Hugging Face - 2024 subset"""
    start_time = time.time()
    # Download the Fineweb 2024 subset dataset from Hugging Face hub and process while streaming
    # WARNING: this is a massive dataset of more than 650GB and will take up to 48 hours to fully process
    dataset = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10",
                       split="train", streaming=True)

    k = 0
    fineweb_file = []  # List to hold pre-processed text
    current_size = 0  # Track the size of the accumulated text
    target_size = 3000 * 1024 # KB size of each file

    print(f'{Fore.GREEN}Beginning FineWeb Dataset Processing...{Fore.RESET}')
    for sample in tqdm(dataset):
        text = sample['text'] + ' [EOD]' # Add 'End of Document' token at the end of each web sample
        added_size = len(text.encode('utf-8'))

        if current_size + added_size > target_size:
            output_file = os.path.join(output_dir, f'fineweb-{k}.txt')
            with open(output_file, 'w', encoding='utf-8') as outfile:
                for doc in fineweb_file:
                    outfile.write(doc + '\n\n')

            k += 1
            fineweb_file = []
            current_size = 0

        fineweb_file.append(text)
        current_size += added_size

    if fineweb_file:
        output_file = os.path.join(output_dir, f'fineweb-{k}.txt')
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for doc in fineweb_file:
                outfile.write(doc + '\n\n')

    # Create train/valid split directories and move processed files into them
    train_valid_splits(output_dir)

    print(f'{Fore.GREEN}FineWeb Dataset Processing Complete!{Fore.RESET}',
        f'\nTotal processing time: {formatted_time(start_time)}')


## ============================= Social Dataset ==============================

def process_social_dataset(data_dir, output_dir, column='Text'):
    start_time = time.time()
    """Process collected social media posts in a dataframe with a 'Text' column"""
    df_files = get_file_paths(data_dir)

    k = 0  # File counter
    social_file = []  # List to hold pre-processed text
    current_size = 0  # Track the size of the accumulated text
    target_size = 2500 * 1024  # Target size in bytes

    print(f'{Fore.GREEN}Beginning Social Media Dataset Processing...{Fore.RESET}')
    for file_name in tqdm(df_files):
        df = pd.read_csv(file_name)
        text_lines = df[column].tolist()

        for line in text_lines:
            if len(line) > 50:
                text = line + ' [EOD]' # Add 'End of Document' token after each social media post
                added_size = len(text.encode('utf-8'))  # Get the size of text in bytes

                # Check if adding this text would exceed the target size
                if current_size + added_size > target_size:
                    output_file = os.path.join(output_dir, f"social-{k}.txt")
                    with open(output_file, 'w', encoding='utf-8') as outfile:
                        for doc in social_file:
                            outfile.write(doc + '\n\n')
                    k += 1
                    social_file = []  # Reset for the next file
                    current_size = 0  # Reset size counter

                # Add the current text after checking for size
                social_file.append(text)
                current_size += added_size

    # Handle any remaining content after the loop
    if social_file:
        output_file = os.path.join(output_dir, f"social-{k}.txt")
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for doc in social_file:
                outfile.write(doc + '\n\n')

    # Create train/valid split directories and move processed files into them
    train_valid_splits(output_dir)

    print(f'{Fore.GREEN}Social Media Dataset Processing Complete!{Fore.RESET}',
        f'\nTotal processing time: {formatted_time(start_time)}')


## ============================== BERT Dataset ===============================

def create_neg_samples(data_dir):
    """Create negative samples for BERT sentence pairs from another dataset."""

    file_paths = get_file_paths(data_dir)
    SPLIT_PUNCT = re.compile(r'([a-zA-Z]+[,.!?]{1,3})([a-zA-Z]+)')
    neg_samples = []

    for file in tqdm(file_paths):
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
            # Remove special token and split on new lines
            text = re.sub(r'\[EOD\]', '', text)
            text_lines = text.split('\n')

        for line in text_lines:
            # Clean up text formating
            line = SPLIT_PUNCT.sub(r'\1 \2', line)
            sentences = sent_tokenize(line)
            for sent in sentences:
                # Keep sentences longer than 15 words
                if len(sent.split()) > 15:
                    neg_samples.append(sent)

    return neg_samples

def process_bert_dataset(neg_dir, data_dir, out_dir):
    """Prepare a BERT dataset from pre-processed pre-training datasets."""
    start_time = time.time()

    print(f'{Fore.GREEN}Beginning BERT Dataset Processing...{Fore.RESET}')
    neg_samples = create_neg_samples(neg_dir)

    neg_idx = 0  # Negative sample counter
    file_idx = 0  # File index counter
    data_file = []  # List to hold sentence pair dicts

    train_files = get_file_paths(data_dir)
    for file_path in tqdm(train_files):
        with open(file_path, 'r', encoding='utf-8') as infile:
            text = infile.read()
            # Simple pre-processing to remove special tokens and
            # Split on \n+ characters to obtain full paragraphs
            text = re.sub(r'\[EOD\]', '', text)
            text_lines = re.split(r'\n+', text)

        for line in text_lines:
            if len(line) > 500: # Drop paragraphs less than 500 bytes/characters
                paragraph = sent_tokenize(line) 
                for i in range(len(paragraph) - 1):
                    # Check if both sentences are long enough
                    if len(paragraph[i].split()) > 10 and len(paragraph[i+1].split()) > 10:
                        if random.random() < 0.5:  # 50% change to be a correct pair
                            data = {'sentence': paragraph[i], 'pair': paragraph[i+1], 'is_next': 1}
                        else:  # 50% change to be a negative / non-matching pair
                            if neg_idx == len(neg_samples):  # Check if the sample list is exhausted
                                neg_idx = 0 # Start over
                            data = {'sentence': paragraph[i], 'pair': neg_samples[neg_idx], 'is_next': 0}
                            neg_idx += 1
                        data_file.append(data)

                    # Save the dataframe with sentence, pair, and is-next value
                    if len(data_file) >= 8000:
                        random.shuffle(data_file)
                        df = pd.DataFrame(data_file)
                        new_file_path = os.path.join(out_dir, f'bert-{file_idx}.csv')
                        df.to_csv(new_file_path, index=False)
                        file_idx += 1 # Increment file index
                        data_file = [] # Reset sentence pair list

    if data_file:
        random.shuffle(data_file)
        df = pd.DataFrame(data_file)
        new_file_path = os.path.join(out_dir, f'bert-{file_idx}.csv')
        df.to_csv(new_file_path, index=False)

    # Create train/valid split directories and move processed files into them
    train_valid_splits(out_dir)

    print(f'{Fore.GREEN}BERT Dataset Processing Complete!{Fore.RESET}',
        f'\nTotal processing time: {formatted_time(start_time)}')


################################ Fine-Tuning ###############################

## =========================== OpenOrca Dataset ============================

def process_openorca_dataset(output_dir):
    """Process OpenOrca dataset for fine-tuning a GPT model on an instruct/response task."""
    start_time = time.time()
    # Punctuation sandwhiched between two words with no whitespace
    SPLIT_PUNCT = re.compile(r'([a-zA-Z]+[.!?]{1,3})([a-zA-Z]+)')
    # Stream the dataset from Hugging Face hub
    dataset = load_dataset("Open-Orca/OpenOrca", split='train', streaming=True)

    k = 0
    prompt_list = []

    print(f'{Fore.GREEN}Beginning OpenOrca Dataset Processing...{Fore.RESET}')
    for i, sample in enumerate(tqdm(dataset)):
        system = sample['system_prompt'] 
        question = sample['question']
        response = sample['response']
        # Modify the sequence below to include system prompt for few-shot prompting
        sequence = '[BOS] ' + question + ' [SEP] ' + response + ' [SEP][EOS]'
        sequence = SPLIT_PUNCT.sub(r'\1 \2', sequence)
        prompt_list.append(sequence)

        # Save a new csv every 8000 rows
        if (i + 1) % 8000 == 0:
            output_file = os.path.join(output_dir, f"openorca-{k}.csv")
            prompt_df = pd.Series(prompt_list)
            prompt_df.to_csv(output_file, index=False)
            k += 1
            prompt_list = []  # Reset for the next group

    # Create train/valid split directories and move processed files into them
    train_valid_splits(output_dir)

    print(f'{Fore.GREEN}OpenOrca Dataset Processing Complete!{Fore.RESET}',
        f'\nTotal processing time: {formatted_time(start_time)}')


## =========================== UltraChat Dataset ============================

def process_ultrachat_dataset(output_dir, tokenizer_path):
    """Process HF ultrachat dataset for fine-tuning a GPT model on a conversational task."""
    start_time = time.time()
    # Stream the dataset from Hugging Face hub
    dataset = load_dataset("stingning/ultrachat", split='train', streaming=True)

    k = 0
    chat_list = []
    tokenizer = Tokenizer.from_file(tokenizer_path)
    max_seq_len = 1024  # Define maximum sequence length

    print(f'{Fore.GREEN}Beginning UltraChat Dataset Processing...{Fore.RESET}')
    for i, sample in enumerate(tqdm(dataset)):
        dialogue = sample['data']
        # Prepare dialogue and check cumulative length
        full_dialogue = ""
        cumulative_len = 0

        for j, part in enumerate(dialogue):
            # Add special tokens
            if j == 0:
                part = '[BOS] ' + part + ' [SEP]'
            else:
                part += ' [SEP]'

            # Check length with tokenizer
            tokens = tokenizer.encode(part).ids
            if cumulative_len + len(tokens) > max_seq_len:
                break  # Stop adding if next part exceeds max sequence length
            cumulative_len += len(tokens)
            full_dialogue += part + " "  # Append part to full dialogue

        if full_dialogue.strip():
            chat_list.append(full_dialogue.strip() + '[EOS]')  # Only append if there's valid data

        # Save a new csv every 8000 rows
        if (i + 1) % 8000 == 0:
            output_file = os.path.join(output_dir, f"ultrachat-{k}.csv")
            df = pd.Series(chat_list)
            df.to_csv(output_file, index=False)
            k += 1
            chat_list = []  # Reset for the next group

    # Handle remaining data
    if chat_list:
        output_file = os.path.join(output_dir, f"ultrachat-{k}.csv")
        df = pd.Series(chat_list)
        df.to_csv(output_file, index=False)

    # Create train/valid split directories and move processed files into them
    train_valid_splits(output_dir)

    print(f'{Fore.GREEN}UltrChat Dataset Processing Complete!{Fore.RESET}',
    f'\nTotal processing time: {formatted_time(start_time)}')


## ======================= BART Translation Datasets ========================

def process_opus100_dataset(output_dir, subset="ar-en", language="arabic"):
    """Process Opus-100 specific language sub-set for machine translation task."""
    # Stream the dataset from HuggingFace hub
    dataset = load_dataset("Helsinki-NLP/opus-100", subset, split='train', streaming=True)

    print(f'{Fore.GREEN}Processing Opus-100 {subset} subset...{Fore.RESET}')

    processed_dataset = []
    for sample in tqdm(dataset):
        # Extract the nested dictionary containing translations
        translation = sample.get('translation', {})
        
        # Ensure both 'en' and the source language exist
        lang = re.sub(r'-?en-?', '', subset)
        if 'en' in translation and lang in translation:
            source = translation[lang]  # Extract source text based on the language code (e.g., 'ar')
            target = translation['en']  # Extract the English translation
        else:
            continue  # Skip if one of the keys is missing
        
        processed_dataset.append({'source': source, 'target': target})

    df = pd.DataFrame(processed_dataset)
    df.to_csv(os.path.join(output_dir, f"{language}.csv"), index=False)

###### Example Usage ######
output = "D:\\Datasets\\fine-tune\\translation"
subset = "en-es"
language = "spanish"
# process_opus100_dataset(output, subset, language)


def split_opus_dataset(data_dir, language, file_size=8000):
    """Reduce a single csv dataset down to smaller file chunks for better data streaming during training."""
    # Load the dataset
    dataset = pd.read_csv(os.path.join(data_dir, f'{language}.csv')).values.tolist()
    random.shuffle(dataset)

    # Create the output directory
    output_dir = os.path.join(data_dir, language)
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

    k = 0
    data_subset = []
    for i, sample in enumerate(tqdm(dataset), 1):
        # Append sample data to the subset
        data_subset.append({'source': sample[0], 'target': sample[1]})
        
        # Save a new csv for every `file_size` sample
        if i % file_size == 0:
            df = pd.DataFrame(data_subset)
            df.to_csv(os.path.join(output_dir, f"{language}-{k}.csv"), index=False)
            data_subset = []  # Clear the subset for the next batch
            k += 1

    # Save any remaining samples that didn't fill up a full batch
    if data_subset:
        df = pd.DataFrame(data_subset)
        df.to_csv(os.path.join(output_dir, f"{language}-{k}.csv"), index=False)

    # Split the dataset into training/validation sets (assuming this function exists)
    train_valid_splits(output_dir)

###### Example Usage ######
data_dir = "D:\\Datasets\\fine-tune\\translation"
# split_opus_dataset(data_dir, "spanish")


## ====================== BART Summarization Datasets =======================

def process_summarization_dataset(output_dir, tokenizer_path, seq_len=1024, file_size=8000):
    """Process HF summary dataset for fine-tuning a BART model on a summarization task."""
    tokenizer = Tokenizer.from_file(tokenizer_path)
    # Stream the dataset from Hugging Face hub
    dataset = load_dataset("jordiclive/scored_summarization_datasets", split='train', streaming=True)

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    k = 0
    processed_dataset = []
    for i, sample in enumerate(tqdm(dataset), 1):
        token_count = sample['t5_text_token_count']
        text = sample['text']
        summary = sample['summary']
        # Filter samples that are less than the model context window size
        if token_count < seq_len:
            token_ids = tokenizer.encode(text).ids
            if len(token_ids) < seq_len:
                processed_dataset.append({'source': text, 'target': summary, 'token_count': len(token_ids)})

        # Save a new csv for every `file_size` sample
        if i % file_size == 0:
            df = pd.DataFrame(processed_dataset)
            df.to_csv(os.path.join(output_dir, f"summary-{k}.csv"), index=False)
            processed_dataset = []  # Clear the subset for the next batch
            k += 1

    # Save any remaining samples that didn't fill up a full batch
    if processed_dataset:
        df = pd.DataFrame(processed_dataset)
        df.to_csv(os.path.join(output_dir, f"summary-{k}.csv"), index=False)

    # Split the dataset into training/validation sets (assuming this function exists)
    train_valid_splits(output_dir)

###### Example Usage ######
output_dir = "D:\\Datasets\\fine-tune\\summarization"
tokenizer_path = "Language/saved_models/bpe-tokenizer.json"
# process_summarization_dataset(output_dir, tokenizer_path)

## ============================== NER Datasets ===============================

def convert_nernews_dataset(input_path, output_path):
    """Convert Conll NER Dataset from this link: 
    https://www.kaggle.com/datasets/namanj27/ner-dataset?resource=download"""

    def convert_ner_tags(tag_list):
        """For ConLL IOB formatted NER tag conversion"""
        converted_tags = []
        for tag in tag_list:
            if tag.endswith('per'):
                tag_type = "PER"
            elif tag.endswith('org'):
                tag_type = "ORG"
            elif tag.endswith('geo'):
                tag_type = "LOC"
            elif tag.endswith('gpe'):
                tag_type = "GPE"
            elif tag.endswith('tim'):
                tag_type = "TIME"
            else:
                tag_type = "MISC"

            if tag.startswith('B-'):
                converted_tags.append(f"B-{tag_type}")
            elif tag.startswith('I-'):
                converted_tags.append(f"I-{tag_type}")
            else:
                converted_tags.append("O")
            
        return converted_tags

    try:
        df = pd.read_csv(input_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(input_path, encoding='ISO-8859-1')  # or 'cp1252' for Windows-1252

    df['Sentence #'] = df['Sentence #'].fillna(method='ffill')

    # Group by 'Sentence #' and aggregate 'Word' and 'Tag' into lists
    grouped = df.groupby('Sentence #').agg({
        'Word': list,
        'Tag': list
    }).reset_index(drop=True)

    # Rename columns for clarity
    grouped.columns = ['tokens', 'ner_tags']
    # Convert ner tag formatting
    grouped['ner_tags'] = grouped['ner_tags'].apply(lambda x: convert_ner_tags(x))

    # Save the formatted DataFrame to a new CSV
    grouped.to_csv(output_path, index=False)


# Function to convert NER ID to tag
def get_tag(tag_id, next_tag_id=None):
    """For MultiNERD dataset NER ID conversion"""
    if tag_id == 0:
        return "O"
    tag_type = "PER" if tag_id in [1, 2] else "ORG" if tag_id in [3, 4] else "LOC" if tag_id in [5, 6] else 'TIME' if tag_id in [27, 28] else "MISC"
    if tag_id % 2 == 1:
        return f"B-{tag_type}" if next_tag_id % 2 == 0 and next_tag_id != 0 else f"S-{tag_type}"
    else:
        return f"I-{tag_type}"
    
def process_hf_ner(output_dir, tokenizer_path, batch_size=10000):
    """Process a NER dataset from HuggingFace Hub for fine-tuning a BERT classification model."""

    tokenizer = Tokenizer.from_file(tokenizer_path)
    dataset = load_dataset("tner/multinerd", 'en', split='test', streaming=True)

    k = 0
    processed_data = []
    # Adjust tags for BERT's subword tokens
    for j, sample in enumerate(tqdm(dataset)):
        if len(sample['tokens']) < 15:
            continue

        wp_tokens = ['[CLS]']
        wp_ner_tags = ['[CLS]'] 
        skip_sample = False

        for i, (word, tag_id) in enumerate(zip(sample['tokens'], sample['tags'])):
            next_id = sample['tags'][i + 1] if i + 1 < len(sample['tags']) else None
            ner_tag = get_tag(tag_id, next_id)
            tokens = tokenizer.encode(word, add_special_tokens=False).tokens

            if not tokens:
                print(f"Skipping sample: {j}")
                skip_sample = True
                break 

            if len(tokens) > 1:
                if not tokens[1].startswith('##'):
                    ner_tag = ner_tag.replace('S-', 'B-')
                wp_tokens.append(tokens[0])
                wp_ner_tags.append(ner_tag)

                for subword in tokens[1:]:
                    wp_tokens.append(subword)
                    if subword.startswith('##'):
                        wp_ner_tags.append(f'##{ner_tag}')
                    else:
                        if ner_tag.startswith('B-'):
                            ner_tag = ner_tag.replace('B-', 'I-')
                        wp_ner_tags.append(ner_tag)
            else:
                wp_tokens.append(tokens[0])
                wp_ner_tags.append(ner_tag)
        
        if skip_sample:
            continue  

        wp_tokens.append('[SEP]')
        wp_ner_tags.append('[SEP]')
        processed_data.append({'tokens': sample['tokens'], 'bert_tokens': wp_tokens, 'tags': wp_ner_tags})

        # Save in batches
        if len(processed_data) >= batch_size:
            output_file = os.path.join(output_dir, f'ner-{k}.csv')
            pd.DataFrame(processed_data).to_csv(output_file, index=False)
            processed_data = []
            k += 1

    # Save any remaining data
    if processed_data:
        output_file = os.path.join(output_dir, f'ner-{k}.csv')
        pd.DataFrame(processed_data).to_csv(output_file, index=False)


############################ NER Encoder/Decoder #############################

def ner_encoder(ner_tags, pad_seq=True, seq_len=512):
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

def ner_decoder(ner_ids, skip_special=True):
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


############################# NER Conll Converter #############################

def convert_conll_file(root_dir, conll_files, file_name):
    reader = ConllCorpusReader(root=root_dir,
                            fileids=conll_files,
                            columntypes=('words', 'pos', 'tree', 'chunk'),
                            separator=' ')

    ner_list = []
    for sent in tqdm(reader.iob_sents()):
        tokens = []
        tags = []
        for word, _, tag in sent:
            if word == "n't":
                if tokens:  # append n't to the previous word
                    tokens[-1] += word
                else:
                    # just in case n't starts a sentence, handle gracefully
                    tokens.append(word)
                    tags.append(tag)
            else:
                tokens.append(word)
                tags.append(re.sub(r'\r', '', tag))

        ner_list.append({'tokens': tokens, 'ner_tags': tags})

    ner_df = pd.DataFrame(ner_list)
    ner_df.to_csv(os.path.join(root_dir, file_name), index=False)

###### Example Usage ######
root_dir = 'C:\\Users\\Tom\\Desktop\\VS Code Projects\\Python Projects\\Label\\data'
conll_files = ['sm1.conll', 'wiki2-sample.conll', 'fineweb1.conll']
file_name = 'ner-combined-conll-tags.csv'
# convert_conll_file(root_dir, conll_files, file_name)




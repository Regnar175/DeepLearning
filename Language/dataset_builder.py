import ast
import torch
from torch.nn.functional import softmax
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from tqdm import tqdm
import pandas as pd


# Global manual seed
torch.manual_seed(42)
torch.cuda.manual_seed(42) 


######################## Text Generation ########################

def generate_chat_dataset(comments, output_path):
    """
    Build a chat dataset using HuggingFace Models for generating relevant questions 
    or responses to a list of comments. 
    Args:
        comments: List of comments from an existing dataset or from social media data.
        output_path: Path to save the generated dataset in csv format.
    """

    # Load the model and tokenizer
    # model = "microsoft/DialoGPT-medium"
    model = "tiiuae/falcon-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model)
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    dataset = []
    for comment in comments:
        if comment.strip().endswith('?'):
            system_prompt = "Respond naturally to the user's question."
        else:
            system_prompt = "Generate a relevant question that might logically precede or relate to the user's comment."

        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": comment},
        ]

        # Generate a response or a new question
        model_output = generator(prompt, max_new_tokens=100)
        generated_text = model_output[0]['generated_text']

        # Arrange the comment/output in the proper order based on its context
        dataset_entry = {
            'prompt': comment if comment.strip().endswith('?') else generated_text,
            'response': generated_text if comment.strip().endswith('?') else comment
        }
        dataset.append(dataset_entry)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(dataset)
    df.to_csv(output_path, index=False)


############################# NER Tagging #############################

def label_ner_dataset(input_path, output_path):
    """
    Build a NER dataset using HuggingFace Models for tagging. Logic includes a loop
    that will convert the tagged entities to a ConLL formated dataset for fine-tuning on NER task.
    Args:
        input_path = path to csv dataset with a 'text' column to tag sentences for training.
        output_path = path to save the newly created dataset in csv format.
    """
    
    TIME = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    model_path = "dslim/bert-large-NER"
    # model_path = "dslim/bert-base-NER"
    # model_path = "dbmdz/bert-large-cased-finetuned-conll03-english"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.to('cuda')

    def get_ner_tags(text):
        # Tokenize the text; return_tensors to 'pt' (PyTorch tensors)
        encoded_inputs = tokenizer(text, return_tensors='pt') 
        encoded_inputs.to('cuda')
        # Get the model's output (predictions)
        with torch.no_grad():
            outputs = model(**encoded_inputs)
        # Decode the predicted tags
        predictions = torch.argmax(outputs.logits, dim=-1)
        tags = [model.config.id2label[p.item()] for p in predictions.squeeze()]

        return tags

    df = pd.read_csv(input_path)
    text_list = df['text'].to_list()

    dataset = []
    for sample in tqdm(text_list):
        tags = get_ner_tags(sample)
        tokens = tokenizer.tokenize(sample, add_special_tokens=True)

        merged_tokens = []
        merged_tags = []
        buffer_token = ""
        current_tag = ""
        # Remove special tokens and merge sub-tokens/tags
        for token, tag in zip(tokens, tags):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                if token.startswith("##"):
                    buffer_token += token[2:]  # Remove '##' and merge
                else:
                    if buffer_token:
                        # Append the previous full token and its tag
                        merged_tokens.append(buffer_token)
                        if buffer_token in TIME:
                            merged_tags.append('B-TIME')
                        else:
                            merged_tags.append(current_tag)
                        buffer_token = ""
                    buffer_token = token  # Start new token
                    # Update current tag
                    if current_tag == 'O' and tag.startswith('I-'):
                        current_tag = tag.replace('I-', 'B-')
                    else:
                        current_tag = tag 

        # Append the last buffered token
        if buffer_token:
            merged_tokens.append(buffer_token)
            merged_tags.append(current_tag)

        dataset.append({'tokens': merged_tokens, 'ner_tags': merged_tags})

    ner_df = pd.DataFrame(dataset)
    ner_df.to_csv(output_path, index=False)


######################## Sentiment Analysis ########################

def label_sentiment_dataset(input_path, output_path, column='text', is_text=True):
    """
    Build a sentiment dataset using HuggingFace Models for generating sentiment scores. 
    Creates three columns with sentiment scores for negative, neutral, and positive values.
    Args:
        input_path = path to csv dataset with text or token columns.
        output_path = path to save the modified dataset in csv format.
        column = title of the column containing text or tokenized sentences.
        is_text = True | if the column is in string format.
    """

    model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to('cuda')

    def get_sentiment_scores(sample):
        if is_text:
            text = sample
        else:
            text = ' '.join(ast.literal_eval(sample))
        # Tokenize the texts; set padding to True and return_tensors to 'pt' (PyTorch tensors)
        encoded_inputs = tokenizer(text, return_tensors='pt') 
        encoded_inputs.to('cuda')
        # Get the model's output (logits)
        with torch.no_grad():
            outputs = model(**encoded_inputs)
        # The logits are located in the logits attribute
        logits = outputs.logits
        # Convert logits to probabilities using softmax
        probabilities = softmax(logits, dim=1)

        return probabilities.squeeze().tolist()

    if isinstance(input_path, list):
        df = pd.concat([pd.read_csv(file) for file in input_path])
    else:
        df = pd.read_csv(input_path)
    
    text_list = df[column].to_list()

    df['negative'] = 0
    df['neutral'] = 0
    df['positive'] = 0

    negative = []
    neutral = []
    positive = []
    for sample in tqdm(text_list):
        scores = get_sentiment_scores(sample)
        negative.append(scores[0])
        neutral.append(scores[1])
        positive.append(scores[2])

    df['negative'] = negative
    df['neutral'] = neutral
    df['positive'] = positive

    df.to_csv(output_path, index=False)

###### Example Usage ######
import os
data_dir = "C:\\Users\\Tom\\Desktop\\VS Code Projects\\Python Projects\\Label\\data"
input_path = [os.path.join(data_dir, "ner-combined-conll-tags.csv"), os.path.join(data_dir, "ner-news1-tags.csv"), os.path.join(data_dir, "ner-sm-tags.csv")]
output_path = "C:\\Users\\Tom\\Desktop\\VS Code Projects\\Python Projects\\DeepLearning\\data\\ner-sent-dataset.csv"

label_sentiment_dataset(input_path, output_path, column='tokens', is_text=False)
import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, processors, decoders


def get_file_paths(data_dir):
    file_paths = []
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(subdir, file)
            file_paths.append(file_path)
    return file_paths

####################### Word Piece Trainer ######################

def train_wp_tokenizer(file_paths, save_path):
    """
    Args: 
        file_paths = list of file paths to similar data used for pre-training.
        save_path = save path to trained tokenizer file.
    """

    trainer = trainers.WordPieceTrainer(vocab_size=50304, show_progress=True, 
                special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"])
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(strip_accents=True)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
    )
    tokenizer.decoder = decoders.WordPiece()

    tokenizer.train(file_paths, trainer)
    tokenizer.save(save_path)

########################## BPE Trainer ##########################

def train_bpe_tokenizer(file_paths, save_path):
    """
    Args: 
        file_paths = list of file paths to similar data used for pre-training.
        save_path = save path to trained tokenizer file.
    """

    trainer = trainers.BpeTrainer(vocab_size=50304, show_progress=True, 
                special_tokens=['[PAD]', '[BOS]', '[EOS]', '[CLS]', '[SEP]', '[EOD]', '[MASK]', '[UNK]'],
                max_token_length=15)
    tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
    tokenizer.normalizer = normalizers.Sequence([
                            normalizers.NFD(), 
                            normalizers.StripAccents(), 
                            normalizers.Lowercase()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()

    tokenizer.train(file_paths, trainer)
    tokenizer.save(save_path)

###################### Word Level Trainer #######################

def train_word_tokenizer(file_paths, save_path):
    """
    Args: 
        file_paths = list of file paths to similar data used for pre-training.
        save_path = save path to trained tokenizer file.
    """

    trainer = trainers.WordLevelTrainer(vocab_size=50304, show_progress=True, 
                special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"])
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([
                            normalizers.StripAccents(), 
                            normalizers.Lowercase()])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
    )

    tokenizer.train(file_paths, trainer)
    tokenizer.save(save_path)


######################## Example Usage #########################

book_files = get_file_paths("D:\\Datasets\\pre-train\\books")
wiki_files = get_file_paths("D:\\Datasets\\pre-train\\wiki")
fineweb_files = get_file_paths("D:\\Datasets\\pre-train\\fineweb")

file_paths = book_files + wiki_files + fineweb_files
save_path = "Language/saved_models/bpe-tokenizer.json"

train_bpe_tokenizer(file_paths, save_path)



                






import os
import sys
import re
import time
import warnings
import torch
from dataclasses import dataclass
from tokenizers import Tokenizer
import pyqtgraph as pg
from PySide6.QtWidgets import QMainWindow, QApplication
from PySide6.QtCore import QThreadPool, Slot, Qt, QEvent
from colorama import init, Fore, Style

from UI.ui_files.MainWindow import Ui_MainWindow
from UI.utils import Signals, ConsoleLogger, Worker, ConfirmDialog, ErrorMessageBox, NoticeMessageBox
from GPT.model import GPTModel
from BERT.model import UncasedClassifier

######################## Global Settings #########################

# Suppress the specific FutureWarning related to torch.load
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")

# Model paths
GPT_ULTRA = os.path.abspath("Language/saved_models/gpt-ultra.pt")
GPT_CHAT = os.path.abspath("Language/saved_models/gpt-uncensored.pt")
GPT_ORCA = os.path.abspath("Language/saved_models/gpt-orca.pt")
# Colorama
init(autoreset=True, convert=False, strip=False)
# Torch manual seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)  
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

######################## Model Parameters ########################
        
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
    load_path = "Language/saved_models/gpt-ultra.pt" # Starting model
    token_path = "Language/saved_models/bpe-tokenizer.json"
    # Special token ids
    bos_id = 1
    eos_id = 2
    cls_id = 3
    sep_id = 4

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

############################ Chat Bot ############################

class ChatBot(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, parent=None, **kwargs):
        super(ChatBot, self).__init__(*args, parent, **kwargs)
        self.setupUi(self)

        # ============== Initialize Supporting QT Objects ================
        self.signals = Signals()
        self.threadpool = QThreadPool().globalInstance()
        self.errorMessage = ErrorMessageBox(self)
        self.noticeMessage = NoticeMessageBox(self)
        self.confirmClear = ConfirmDialog("Clear Console", "Clear console chat history?", parent=self)
        # Redirect sys.stdout to the console logger
        self.logger = ConsoleLogger(self.console)
        sys.stdout = self.logger
        # Model combo box
        models = ['GPT Ultra Chat', 'GPT Uncensored Chat', 'GPT Instruct']
        self.modelBox.addItems(models)
        self.modelBox.setCurrentIndex(0)
        # Temp combo box
        temp = ['Temp', '0.9', '0.8', '0.7', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1']
        self.tempBox.addItems(temp)
        self.tempBox.setCurrentIndex(0)
        # Top K combo box
        top_k = ['Top K', '25', '50', '75', '100', '150', '200', '300', '400', '500']
        self.topkBox.addItems(top_k)
        self.topkBox.setCurrentIndex(0)
        # Top P combo box
        top_p = ['Top P', '0.9', '0.8', '0.7', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1']
        self.toppBox.addItems(top_p)
        self.toppBox.setCurrentIndex(0)

        # ============== Connect Widget and Object Signals ==============
        self.actionQuit.triggered.connect(self.quit_bot)
        self.loadButton.clicked.connect(self.load_model)
        self.clearButton.clicked.connect(self.clear_console)
        self.textBox.installEventFilter(self)
        self.signals.enterPressed.connect(self.submit_prompt)

        # =================== Initialize LLM Objects ====================
        self.gpt_config = GPTConfig()
        self.gpt_model = GPTModel.from_pretrained(self.gpt_config)
        self.tokenizer = Tokenizer.from_file(self.gpt_config.token_path)
        self.bert_config = BERTConfig()
        self.classifier = UncasedClassifier(self.bert_config)

    @Slot()
    def quit_bot(self):
        """Closes the chat bot via the menu drop down"""
        self.close()

    @Slot()
    def load_model(self):
        """Method connected to the loadButton clicked signal.\n
        Loads the selected model from the drop down."""
        # Select the model
        if self.modelBox.currentIndex() == 0:
            self.gpt_config.load_path = GPT_ULTRA
            model = "GPT Ultra Chat"
        elif self.modelBox.currentIndex() == 1:
            self.gpt_config.load_path = GPT_CHAT
            model = "GPT Uncensored Chat"
        elif self.modelBox.currentIndex() == 2:
            self.gpt_config.load_path = GPT_ORCA
            model = "GPT Instruct"
        
        # Load the new model weights
        self.gpt_model = GPTModel.from_pretrained(self.gpt_config)
        # Notify the model is loaded
        self.noticeMessage.notice('Model Loaded', f'{model} model is loaded and ready!')

    @Slot()
    def clear_console(self):
        """Method connected to the clearButton clicked signal.\n
        Clear the console of its chat history and clears the text box."""
        confirm = self.confirmClear # Custom confirm clear dialog
        if confirm.exec():
            self.console.clear()
            self.textBox.clear()
            self.promptLabel.setText('')
            self.respLabel.setText('')
            self.timeLabel.setText('')
            self.wordsLabel.setText('')
            self.scoreLabel.setText('')
            print(f"{Fore.GREEN}Console chat history and all metrics have been cleared{Fore.RESET}!")
            print()

    @Slot()
    def submit_prompt(self, text):
        """Enter a question and submit the prompt to the model via the threadpool worker"""
        print(f"<font color=\"#FF5500\">USER:</font> {text}")
        print()
        # Create a worker object
        worker = Worker(self.inference, text) 
        # Connect the worker signals
        worker.signals.result.connect(self.model_output)
        worker.signals.exception.connect(self.threadpool_exception)
        # Start the worker
        self.threadpool.start(worker)

    @Slot()
    def model_output(self, result):
        gen_time = result['time']
        similarity = result['similarity']
        # Display the response
        print(f"{Fore.BLACK}CHAT BOT:{Fore.RESET} {result['resp_text']}")
        print()
        # Update the display widget values
        self.promptLabel.setText(self.get_polarity(result['prompt_sent']))
        self.respLabel.setText(self.get_polarity(result['resp_sent']))
        self.timeLabel.setText(f'{gen_time:.2f} secs')
        self.wordsLabel.setText(result['num_words'])
        self.scoreLabel.setText(f'{similarity:.2f}%')

    @Slot()
    def threadpool_exception(self):
        """Slot to display the error message box in the main thread."""
        self.errorMessage.critical("Threadpool Exception!", "Exception! See error log for details.")

    def get_polarity(self, sentiment):
        """Determine polarity based on the sentiment score."""
        # Extract sentiment scores and convert to percentage
        neg_score = sentiment['negative'] * 100
        neu_score = sentiment['neutral'] * 100
        pos_score = sentiment['positive'] * 100
        # Format the display text
        negative = f'<font color=\"#AA0000\">{neg_score:.1f}% NEG</font> |'
        neutral = f' <font color=\"#00AA00\">{neu_score:.1f}% NEU</font> |'
        positive = f' <font color=\"#00AAFF\">{pos_score:.1f}% POS</font>'

        return negative + neutral + positive

    def inference(self, input_text):
        """Submit the user prompt to the model and return the formatted output"""
        if self.tagCheck.isChecked() == True:
            color = True
        else:
            color = False

        if self.tempBox.currentIndex() == 0:
            temp = 1.0
        else:
            temp = float(self.tempBox.currentText())

        if self.topkBox.currentIndex() == 0:
            top_k = None
        else:
            top_k = int(self.topkBox.currentText())

        if self.toppBox.currentIndex() == 0:
            top_p = None
        else:
            top_p = float(self.toppBox.currentText())
        
        start_time = time.time()
        # Encode the prompt text
        prompt_ids = self.tokenizer.encode(input_text).ids
        prompt = [1] + prompt_ids + [4] # Insert [BOS] and [SEP] special tokens
        # Generate the response and decode the output
        generated_ids = self.gpt_model.generate(prompt, max_length=1024, temp=temp, top_k=top_k, top_p=top_p)
        generated_text = self.tokenizer.decode(generated_ids[len(prompt):])
        # Format (tag) the prompt and response text and return sentiment scores for each
        _, prompt_sent, _ = self.classifier(input_text, color=color)
        resp_text, resp_sent, _ = self.classifier(generated_text, color=color)
        # Get the cosine similarity between the prompt and generated text
        response_ids = self.tokenizer.encode(generated_text).ids
        similarity = self.gpt_model.get_similarity(prompt_ids, response_ids)
        # Calculate the total model inference time
        process_time = time.time() - start_time

        return {'prompt_sent': prompt_sent,
                'resp_text': resp_text,
                'resp_sent': resp_sent,
                'num_words': str(len(resp_text.split())),
                'similarity': similarity * 100,
                'time': process_time,
                }

    def eventFilter(self, obj, event):
        """Custom behavior for the QTextEdit textBox to trigger events upon hitting 'enter'"""
        if obj == self.textBox and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not (event.modifiers() & Qt.ControlModifier):
                self.signals.enterPressed.emit(self.textBox.toPlainText())  # Emit signal with the text
                self.textBox.clear()  # Clear the text edit
                return True  # Indicate that the event has been handled
        return super().eventFilter(obj, event)  

    def closeEvent(self, event):
        """Shuts down application on close, resets stdout / stderr
        to system defaults and clears the threadpool."""
        # Return stdout and stderr to defaults and clears the threadpool
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        self.threadpool.clear()

        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ChatBot()
    window.show()
    sys.exit(app.exec())
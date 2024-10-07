import os
import math
import time
import datetime
import warnings
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler, CyclicLR
from torch.nn.utils import clip_grad_norm_
from torch.amp import GradScaler, autocast
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
from tokenizers import Tokenizer
from colorama import Fore

# Suppress the specific FutureWarning related to torch.load
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")

plt.style.use('dark_background')

# Global manual seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)  
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn


def formatted_time(start_time):
    """Formats processing time."""
    elapsed_time = time.time() - start_time
    # Calculate hours, minutes and seconds
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    return f"{Fore.CYAN}{hours}{Fore.RESET} hrs {Fore.CYAN}{minutes}{Fore.RESET} mins {Fore.CYAN}{seconds:.3f}{Fore.RESET} secs"


class CosineWarmupLR(_LRScheduler):
    """Custom learning rate schheduler that mimics PyTorch's OCLR scheduler,
    but allows for setting a minimum learning rate."""
    def __init__(self, optimizer, warmup_iters, max_iters, max_lr, min_lr, last_epoch=-1):
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.max_lr = max_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            return [self.max_lr * self.last_epoch / self.warmup_iters for _ in self.base_lrs]
        elif self.last_epoch > self.max_iters:
            return [self.min_lr for _ in self.base_lrs]
        else:
            # Cosine decay
            decay_ratio = (self.last_epoch - self.warmup_iters) / (self.max_iters - self.warmup_iters)
            coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
            return [self.min_lr + coeff * (self.max_lr - self.min_lr) for _ in self.base_lrs]


class GradualDecayLR(_LRScheduler):
    """Custom learning rate schheduler that gradually decays the learning rate 
    to a minimum based on total number of training steps."""
    def __init__(self, optimizer, decay_steps, start_lr, min_lr, last_epoch=-1):
        self.decay_steps = decay_steps
        self.start_lr = start_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.decay_steps:
            return [self.min_lr for _ in self.base_lrs]
        else:
            decay_ratio = self.last_epoch / self.decay_steps
            return [self.start_lr - (self.start_lr - self.min_lr) * decay_ratio for _ in self.base_lrs]
        

class Loader:
    """
    Base loader class for pre-training and fine-tuning.
    """
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.single_file = None
        self.tokenizer = Tokenizer.from_file(config.token_path)
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.bos_id = None
        self.eos_id = None
        self.batch_size = config.nbatch
        self.seq_len = config.seqlen
        self.batch_len = (config.nbatch * config.seqlen) - config.nbatch
        self.epochs = config.epochs
        self.epoch_count = 0
        self.train_files = self.get_file_paths(os.path.join(config.data_dir, 'train'))
        self.valid_files = self.get_file_paths(os.path.join(config.data_dir, 'valid'))
        self.train_data = []
        self.valid_data = []
        self.train_file_idx = 0
        self.train_data_idx = 0
        self.valid_file_idx = 0
        self.valid_data_idx = 0
        self.num_replicas = 1 # Default 1 GPU


    def get_file_paths(self, data_dir):
        """Create a list of dataset file paths from a given file directory."""
        file_paths = []
        for subdir, dirs, files in os.walk(data_dir):
            for file in files:
                file_path = os.path.join(subdir, file)
                file_paths.append(file_path)
        return file_paths
    
    def set_num_replicas(self, num_replicas):
        """Set num_replicas attribute for total number of GPUs."""
        self.num_replicas = num_replicas

    def prepare_data(self, file_path):
        """Prepare a data file and create a list of batches to call from."""
        pass

    def load_train_data(self):
        """Load prepared data into train_data attribute to get batches from."""
        # Loop back around to the start of the data files if at the end
        if self.train_file_idx >= len(self.train_files):
            self.train_file_idx = 0
            self.train_data_idx = 0
            self.epoch_count += 1
            print(f"{Fore.MAGENTA}Epoch: {Fore.RESET}{Fore.CYAN}{self.epoch_count}{Fore.RESET}")

        file_path = self.train_files[self.train_file_idx]
        self.train_data = self.prepare_data(file_path)

    def load_valid_data(self):
        """Load prepared data into valid_data attribute to get batches from."""
        # Loop back around to the start of the data files if at the end
        if self.valid_file_idx >= len(self.valid_files):
            self.valid_file_idx = 0
            self.valid_data_idx = 0

        file_path = self.valid_files[self.valid_file_idx]
        self.valid_data = self.prepare_data(file_path)
    
    def prepare_batch(self, batch):
        """Prepare batches to be called during training and validation steps."""
        pass
    
    def to_tensors(self, batch):
        """Convert batch to tensors."""
        pass

    def get_train_batch(self):
        """Get the next train batch from the currently loaded train_data list."""
        # Load the next data file if at the end of the current one
        if self.train_data_idx >= len(self.train_data):
            self.train_file_idx += 1
            self.train_data_idx = 0
            self.load_train_data()

        batch = self.train_data[self.train_data_idx]

        self.train_data_idx += 1
        return self.prepare_batch(batch)
    
    def get_valid_batch(self):
        """Get the next train batch from the currently loaded train_data list."""
        # Load the next data file if at the end of the current one
        if self.valid_data_idx >= len(self.valid_data):
            self.valid_file_idx += 1
            self.valid_data_idx = 0
            self.load_valid_data()

        batch = self.valid_data[self.valid_data_idx]

        self.valid_data_idx += 1
        return self.prepare_batch(batch)
    
    def __len__(self):
        # Estimated total number of train batches in the dataset
        return len(self.train_data) * len(self.train_files)

    def __next__(self):
        return self.get_valid_batch()
    
    def __iter__(self):
        """Yield a prepared batch per iteration and cycle through all training data
        files until the desired epoch has finished or the max number of iterations
        has completed in the training loop."""

        while True:
            if self.epochs is not None and self.epoch_count >= self.epochs:
                print(f"{Fore.MAGENTA}Epochs completed: {Fore.RESET}{Fore.CYAN}{self.epoch_count}{Fore.RESET}")
                break  # Stop iterating if the specified number of epochs is reached

            yield self.get_train_batch()


class Trainer:
    """
    Base trainer class for pre-training or fine-tuning on large (multi-file) text/csv file datasets.
    """
    def __init__(self, config):
        self.config = config
        self.ddp = False
        self.model = object()
        self.optimizer = object()
        self.scheduler = None
        self.scaler = GradScaler('cuda') 
        self.loader = object()
        self.iteration = 1 # Starting position
        self.max_iters = config.max_iters
        self.grad_accum_steps = config.grad_accum
        self.train_data_dir = config.data_dir
        self.saved_data_dir = "" # To hold saved dataset directory from last session
        self.train_loss = float('inf')
        self.current_lr = float('inf')
        self.results = {
            'train_loss': [],
            'eval_loss': [],
            'accuracy': [],
            'f1_score': [],
            'perplexity': [],
            'learn_rate': [] 
            }
        

    def init_distributed(self):
        """Initialize distributed training environment."""
        self.rank = int(os.environ['RANK'])  # Rank of this process
        self.world_size = int(os.environ['WORLD_SIZE'])  # Total number of processes
        # Set local rank by calculating mod with GPUs per node
        self.local_rank = self.rank % torch.cuda.device_count()  # GPU index on local machine
        ## Initialize using TCPStore 
        # store = dist.TCPStore(
        #     host_name=os.environ['MASTER_ADDR'], 
        #     port=int(os.environ['MASTER_PORT']),
        #     world_size=self.world_size,
        #     is_master=(self.rank == 0),  # True for the master node
        #     timeout=datetime.timedelta(seconds=300),  # Increase if timeout happens
        #     use_libuv=False
        # )
        # Initialize process group
        dist.init_process_group(
            # backend='nccl' if dist.is_nccl_available() else 'gloo',
            backend='gloo', # init_method='env://',
            init_method=f'tcp://{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}',
            world_size=self.world_size, rank=self.rank,
            # store=store
        )
        # Wrap model with DDP
        self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
        # Set number of GPUs for distributed batch preparation
        self.loader.set_num_replicas(self.world_size)

    def cleanup(self):
        """Cleanup the distributed process group."""
        dist.destroy_process_group()

    def set_scheduler(self, sched):
        if sched == 'warmup':
            self.scheduler = CosineWarmupLR(self.optimizer, warmup_iters=self.config.wu_iters, max_iters=self.config.sched_iters,
                                            max_lr=self.config.max_lr, min_lr=self.config.min_lr)
        elif sched == 'cyclic':
            self.scheduler = CyclicLR(self.optimizer, base_lr=self.config.min_lr, max_lr=self.config.max_lr, mode="triangular2",
                                    step_size_up=self.config.step_up, step_size_down=self.config.step_dn, cycle_momentum=False)
        elif sched == 'decay':
            self.scheduler = GradualDecayLR(self.optimizer, decay_steps=self.config.sched_iters, start_lr=self.config.max_lr,
                                            min_lr=self.config.min_lr)
        elif sched == 'rlrp':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9, min_lr=self.config.min_lr)
        else:
            self.scheduler = None

    def load_state(self, state=None):
        """Load saved model from training checkpoint, load just the weights or start fresh"""
        if state == 'checkpoint':
            print(f"{Fore.MAGENTA}Loading saved model weights with dataset checkpoint...{Fore.RESET}")
            self.load_checkpoint()
        elif state == 'weights':
            print(f"{Fore.MAGENTA}Loading saved model weights with new training dataset...{Fore.RESET}")
            self.load_weights()
            # Set all counters to zero for new dataset
            self.loader.train_file_idx = 0
            self.loader.train_data_idx = 0
            self.loader.valid_file_idx = 0
            self.loader.valid_data_idx = 0
        else: # Start fresh on a new model
            print(f"{Fore.MAGENTA}Training a model from scratch...{Fore.RESET}")
        self.print_checkpoint_info()

    def load_checkpoint(self):
        checkpoint = torch.load(self.config.load_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loader.train_file_idx = checkpoint['train_file_idx']
        self.loader.train_data_idx = checkpoint['train_data_idx']
        self.loader.valid_file_idx = checkpoint['valid_file_idx']
        self.loader.valid_data_idx = checkpoint['valid_data_idx']
        self.iteration = checkpoint['iteration']
        self.train_loss = checkpoint['train_loss']
        self.current_lr = checkpoint['learning_rate']
        self.saved_data_dir = checkpoint['data_directory']

    def load_weights(self):
        model_state = torch.load(self.config.load_path)
        self.model.load_state_dict(model_state['model_state_dict'])
        self.optimizer.load_state_dict(model_state['optimizer_state_dict'])
        self.iteration = model_state['iteration']
        self.train_loss = model_state['train_loss']
        self.current_lr = model_state['learning_rate']
        self.saved_data_dir = model_state['data_directory']

    def save_checkpoint(self, current_loss, current_lr, best_model):
        # Update the state dict
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_file_name': self.config.data_dir,
            'train_file_idx': self.loader.train_file_idx,
            'train_data_idx': self.loader.train_data_idx,
            'valid_file_idx': self.loader.valid_file_idx,
            'valid_data_idx': self.loader.valid_data_idx,
            'iteration': self.iteration,
            'train_loss': current_loss,
            'learning_rate': current_lr,
            'data_directory': self.train_data_dir
        }
        self.current_lr = current_lr
        # Save the checkpoint only if it's the best loss so far
        if best_model:
            # Check if current train loss is lower than the previous best
            is_best = current_loss < self.train_loss
            if is_best:
                self.train_loss = current_loss
                torch.save(state, self.config.save_path)
        else: # Save every 100 training iterations
            self.train_loss = current_loss
            torch.save(state, self.config.save_path)

    def print_checkpoint_info(self, saved_dir=True):
        if saved_dir:
            directory = self.saved_data_dir
        else:
            directory = self.train_data_dir
        print(f"{Fore.GREEN}Training Directory:{Fore.RESET} {directory}")
        print(f"{Fore.GREEN}Iteration Position:{Fore.RESET} {Fore.CYAN}{self.iteration}{Fore.RESET}")
        print(f"{Fore.GREEN}Training Loss:{Fore.RESET} {Fore.CYAN}{self.train_loss:.3f}{Fore.RESET}",
                f"{Fore.GREEN}| Learning Rate:{Fore.RESET} {Fore.CYAN}{self.current_lr:.10f}{Fore.RESET}")
        print(f"{Fore.GREEN}Training File Index:{Fore.RESET} {Fore.CYAN}{self.loader.train_file_idx}{Fore.RESET}",
                f"{Fore.GREEN}| Dataset Index:{Fore.RESET} {Fore.CYAN}{self.loader.train_data_idx}{Fore.RESET}")
        print(f"{Fore.GREEN}Validation File Index:{Fore.RESET} {Fore.CYAN}{self.loader.valid_file_idx}{Fore.RESET}",
                f"{Fore.GREEN}| Dataset Index:{Fore.RESET} {Fore.CYAN}{self.loader.valid_data_idx}{Fore.RESET}")
        
    def print_progress(self, tloss, vloss, acc, f1, perp, lr):
        print(f'Step: {Fore.CYAN}{self.iteration}{Fore.RESET}',
            f"| Train Loss: {Fore.CYAN}{tloss:.3f}{Fore.RESET}",
            f"| Eval Loss: {Fore.CYAN}{vloss:.3f}{Fore.RESET}",
            f"| Acc: {Fore.CYAN}{acc:.3f}{Fore.RESET}",
            f"| F1: {Fore.CYAN}{f1:.3f}{Fore.RESET}",
            f"| Perplex: {Fore.CYAN}{perp:.2f}{Fore.RESET}",
            f"| LR: {Fore.CYAN}{lr:.10f}{Fore.RESET}")

    def plot_results(self):
        """Plots results of training session."""
        train_loss = self.results['train_loss']
        eval_loss = self.results['eval_loss']
        accuracy = self.results['accuracy']
        f1_score = self.results['f1_score']
        perplexity = self.results['perplexity']
        learn = self.results.get('learn_rate', [0]*len(train_loss))
        iters = range(len(train_loss))

        # Set up plots in a 2x2 grid
        plt.figure(figsize=(12, 10))
        # Plot loss
        plt.subplot(2, 2, 1)  
        plt.plot(iters, train_loss, label='Train')
        plt.plot(iters, eval_loss, label='Validation')
        plt.title('Loss')
        plt.xlabel('Iterations x 100')
        plt.ylabel('Loss')
        plt.legend()

        # Plot accuracy / F1
        plt.subplot(2, 2, 2)  
        plt.plot(iters, accuracy, label='Accuracy')
        plt.plot(iters, f1_score, label='F1 Score')
        plt.title('Accuracy / F1')
        plt.xlabel('Iterations x 100')
        plt.ylabel('Acc / F1')
        plt.legend()

        # Plot perplexity
        plt.subplot(2, 2, 3)  
        plt.plot(iters, perplexity, c='m', label='Perplexity')
        plt.title('Perplexity')
        plt.xlabel('Iterations x 100')
        plt.ylabel('Perplexity')
        plt.legend()

        # Plot learning rate
        plt.subplot(2, 2, 4)  
        plt.plot(iters, learn, c='c', label='Learn Rate')
        plt.title('Learning Rate')
        plt.xlabel('Iterations x 100')
        plt.ylabel('Learning Rate')
        plt.legend()

        plt.tight_layout()  
        plt.show()

    def get_accuracy(self, logits, target):
        preds = torch.softmax(logits, dim=-1).argmax(dim=-1)
        mask = (target != 0)  
        valid_preds = preds[mask]
        valid_targets = target[mask]
        correct_predictions = (valid_preds == valid_targets)
        accuracy = correct_predictions.float().mean().item() if valid_targets.numel() > 0 else float('nan')
        return accuracy

    def calculate_f1(self, logits, target):
        preds = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).flatten()
        target = target.flatten()
        mask = (target != 0)  
        valid_preds = preds[mask]
        valid_targets = target[mask]
        # Calculate F1 score using sklearn for simplicity
        return f1_score(valid_targets.cpu().numpy(), valid_preds.cpu().numpy(), average='weighted', zero_division=1)
    
    def train_step(self, batch):
        """Train step using Autocast and Gradscaler. Performs N number of gradient
        accumulation steps before stepping the optimizer and scheduler."""
        pass

    def valid_step(self):
        """Validation step every 100 iterations. Uses Autocast and does the same number 
        of grad scale iterations to remain consistent with the training loss output."""
        pass

    def train(self, save_cp=True, best_model=False, plot_results=True):
        """Model training loop with a validation step every 100 train iterations.\n
        Args:
            save_cp: bool = True | save model and training dataset checkpoint every 100 iterations\n
            best_model: bool = False | only save if it is the best training loss recorded so far\n
            plot_results: bool = True | plot results after training has stopped""" 
        pass

    def test(self):
        """Test the model's performance after a training session."""
        pass



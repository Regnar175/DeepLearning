import os
import time
import math
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.amp import GradScaler, autocast
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from colorama import Fore

from GPT.model import GPTModel
from GPT.loader import GPTPreTrainLoader, GPTFineTuneLoader
from utilities import Trainer, formatted_time


plt.style.use('dark_background')

# Global manual seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)  
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn


class GPTTrainer(Trainer):
    """
    Custom trainer class for pre-training or fine-tuning on large (multi-file) text/csv file datasets.\n
    Params `config`:\n 
        device: 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype: set to bfloat16 or float16 for autocasting dtype
        ntoken: size of tokenizer vocabulary 
        seqlen: length of token sequence per mini-batch
        nbatch: number of mini-batches (token sequences)
        dmodel: embedding dimension size
        nhead: number of attention heads 
        nlayer: number of encoder/decoder layers
        shead: dmodel // nhead = individual head size
        dropout: % of nodes turned off during training
        bias: optionally turn bias on/off in linear/normalization layers
        data_dir: directory path to train/validation datasets
        load_path: path to load model checkpoint/weights
        save_path: path to save model checkpoint/weights 
        lr_sched: LR scheduler type - warmup, cyclic, decay, rlrp, or None
        grad_accum: Gradient accumulation steps to simulate larger batch sizes
        max_iters: Max training iterations
        wu_iters: Warmup iterations for warmup scheduler
        sched_iters: Scheduler iterations - adjust for gradient accumulation
        max_lr: Max or starting learning rate
        min_lr: Minimum learning rate
        step_up: Step up iterations for CLR scheduler
        step_dn: Step down iterations for CLR scheduler
        epochs: number of epochs for the training dataset
    Params `state`:\n
        None = start training a fresh model\n
        'checkpoint' = load model/optimizer state_dicts to continue training on the same dataset\n
        'weights' = load model/optimizer state_dicts to train on a new dataset\n
    Params `pretrain`:\n
        True = default for setting the pre-train lodader class\n 
    Params `distribued`:\n
        False = default for single GPU training\n 
    Attributes:\n
        ddp: bool = False | Distributed training.
        model: object = GPTModel()
        optimizer: object = model.config_optimizer | AdamW optimizer
        scheduler: object = None | config setting to select the learning rate scheduler
        scaler: object = GradScaler() | PyTorch gradient scaler implementation
        loader: object = GPTPreTrainingLoader() | Custom dataloader class
        iteration: int = 1 | training step counter
        max_iters: int = Any | config setting for max training iterations
        grad_accum_steps: int = Any | config setting for gradient accumulation steps
        train_data_dir: str = Any | config setting for training directory path
        saved_data_dir: str = Any | attribute to hold directory of last training session
        train_loss: float = 'inf' | attribute to hold training loss value during training
        current_lr: float = 'inf' | attribute to hold current learning rate during training
        results: dict = Any | dictionary to hold training metric results for plotting
    """
    def __init__(self, config, state=None, pretrain=True, distributed=False):
        super().__init__(config)
        self.config = config
        self.pretrain = pretrain
        self.ddp = distributed
        self.model = GPTModel(config).to(config.device)
        self.optimizer = self.model.config_optimizer(lr=config.max_lr)
        self.scheduler = None
        self.scaler = GradScaler('cuda') 
        if self.pretrain:
            self.loader = GPTPreTrainLoader(config)
        else:
            self.loader = GPTFineTuneLoader(config)
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
        
        self.set_scheduler(config.lr_sched)
        self.load_state(state)
        if self.ddp:
            self.init_distributed()


    def train_step(self, batch):
        """Train step using Autocast and Gradscaler. Performs N number of gradient
        accumulation steps before stepping the optimizer and scheduler."""

        self.model.train()
        # Mixed precision training with torch.amp.autocast()
        # Automatically cast dtype during training
        with autocast(device_type=self.config.device, dtype=self.config.dtype): 
            if self.pretrain:
                mask = None
            else:
                mask = batch['attn_mask']
            # Forward pass
            logits, loss = self.model(batch['input_ids'], batch['target'], mask=mask)
            scaled_loss = loss / self.grad_accum_steps
        # Accumulate gradients    
        self.scaler.scale(scaled_loss).backward()

        if self.iteration % self.grad_accum_steps == 0:
            # Optimizer step using Gradscaler to simulate training with larger batches
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # Flush gradients after scaler/optimizer step
            self.optimizer.zero_grad(set_to_none=True)
            # Step the LR scheduler if available
            if self.scheduler is not None:
                if self.config.lr_sched != 'rlrp':
                    self.scheduler.step()

        return loss.item()
    
    def valid_step(self):
        """Validation step every 100 iterations. Uses Autocast and does the same number 
        of gradient scale iterations to remain consistent with the training loss output."""

        valid_loss, accuracy, f1_score, perplex = 0, 0, 0, 0
        self.model.eval()
        with torch.inference_mode():
            for _ in range(self.grad_accum_steps):
                batch = next(self.loader) # Calls get_valid_batch() via __next__
                if self.pretrain:
                    mask = None
                else:
                    mask = batch['attn_mask']
                # Use autocast with validation step to remain consistent in loss calculation
                with autocast(device_type=self.config.device, dtype=self.config.dtype):
                    # Forward pass
                    logits, loss = self.model(batch['input_ids'], batch['target'], mask=mask)

                if self.config.lr_sched == 'rlrp': # Reduce LR on plateau must step after validation
                    self.scheduler.step(loss)

                # Update validation trackers
                valid_loss += loss.item()
                accuracy += self.get_accuracy(logits, batch['target'])
                f1_score += self.calculate_f1(logits, batch['target'])
                perplex += torch.exp(loss).item()

        return {'loss': valid_loss, 
                'accuracy': accuracy, 
                'f1_score': f1_score,
                'perplex': perplex
                }

    def train(self, save_cp=True, best_model=False, plot_results=True):
        """Model training loop with a validation step every 100 train iterations.\n
        Args:
            save_cp: bool = True | save model and training dataset checkpoint every 100 iterations\n
            best_model: bool = False | only save if it is the best training loss recorded so far\n
            plot_results: bool = True | plot results after training has stopped"""

        print(f"{Fore.MAGENTA}Training Directory for this session:{Fore.RESET} {self.train_data_dir}")
        option = input(f'{Fore.YELLOW}Do you want to begin training (y or n)?:{Fore.RESET} ')
        if option == 'y':
            print(f"{Fore.MAGENTA}Begin training session...{Fore.RESET}")
        else:
            print(f"{Fore.RED}Abort training...{Fore.RESET}")
            return
        
        if self.ddp:
            # Synchronize all processes before training
            dist.barrier()
        # Accumulate train loss per iteration
        train_loss = 0
        start_time = time.time()
        for i, batch in enumerate(self.loader, 1): # Calls get_train_batch() iteratively via __iter__

            if i == self.max_iters:
                print(f"{Fore.MAGENTA}Training complete!{Fore.RESET}")
                self.print_checkpoint_info(saved_dir=False)
                break
            
            train_loss += self.train_step(batch)

            # Validation step every 100 training iterations
            if self.iteration % 100 == 0:
    
                metrics = self.valid_step()

                # Calculate average loss, accuracy, f1, perplixity and log the results
                train_loss /= 100
                valid_loss = metrics['loss'] / self.grad_accum_steps
                accuracy = metrics['accuracy'] / self.grad_accum_steps
                f1_score = metrics['f1_score'] / self.grad_accum_steps
                perplex = metrics['perplex'] / self.grad_accum_steps
                
                # Get the current learning rate and print a status update
                current_lr = self.optimizer.param_groups[0]['lr']
                self.print_progress(train_loss, valid_loss, accuracy, f1_score, perplex, current_lr)
                if self.iteration % 1000 == 0:
                    print(f'Elapsed Training Time: {formatted_time(start_time)}')

                # Update results dictionary for plotting
                self.results['train_loss'].append(train_loss)
                self.results['eval_loss'].append(valid_loss)
                self.results['accuracy'].append(accuracy)
                self.results['f1_score'].append(f1_score)
                self.results['perplexity'].append(perplex)
                self.results['learn_rate'].append(current_lr)

                if save_cp:
                    self.save_checkpoint(train_loss, current_lr, best_model)
                else: # Update train loss and LR trackers if not saving best model
                    self.train_loss = train_loss
                    self.current_lr = current_lr

                # Reset train loss tracker
                train_loss = 0
            # Increment the step counter
            self.iteration += 1
        
        if self.ddp: 
            self.cleanup()
        if plot_results:
            self.plot_results()

    def test(self, text=None):
        """Test the model's performance after a training session."""
        self.model.eval()
        with torch.inference_mode():
            batch = self.loader.get_valid_batch()
            if self.pretrain:
                mask = None
            else:
                mask = batch['attn_mask']
            logits, loss = self.model(batch['input_ids'], batch['target'], mask=mask)

        accuracy = self.get_accuracy(logits, batch['target'])
        f1_score = self.calculate_f1(logits, batch['target'])
        perplexity = torch.exp(loss)

        print(f"{Fore.GREEN}Model Evaluation:{Fore.RESET}")
        print(f"Test Loss: {Fore.CYAN}{loss:.3f}{Fore.RESET}",
            f"| Accuracy: {Fore.CYAN}{accuracy:.3f}{Fore.RESET}",
            f"| F1 Score: {Fore.CYAN}{f1_score:.3f}{Fore.RESET}",
            f"| Perplexity: {Fore.CYAN}{perplexity:.2f}{Fore.RESET}")

        if text is None:
            text = "Let's demonstrate how the model generates text."
        print(f'{Fore.GREEN}Prompt:{Fore.RESET} {text}')
        encoded = self.loader.tokenizer.encode(text).ids

        # Insert [BOS] token to shift tokens right for prediction
        prompt = [self.loader.bos_id] + encoded
        if not self.pretrain:
            prompt += [self.loader.sep_id] # Add [SEP] token at the end to trigger zero-shot prompt/response
        generated_ids = self.model.generate(prompt, max_length=256, temp=1.0, top_k=None)
        generated_text = self.loader.tokenizer.decode(generated_ids[len(prompt):]) # Cut off the prompt ids

        print(f'{Fore.GREEN}Response:{Fore.RESET} {generated_text.strip()}')
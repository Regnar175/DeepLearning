import time
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.amp import GradScaler, autocast
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from colorama import Fore

from BERT.model import BERTModel, BERTClassifier
from BERT.loader import BERTPreTrainLoader, BERTClassifierLoader
from utilities import Trainer, formatted_time

plt.style.use('dark_background')

# Global manual seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)  
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn


class BERTPreTrainer(Trainer):
    """
    Custom trainer class for pre-training on large (multi-file) csv file datasets.\n
    Params `config`:\n 
        device: 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype: set to bfloat16 or float16 for autcasting dtype
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
    Attributes:\n
        ddp: bool = False | Distributed training.
        model: object = BERTModel()
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
    def __init__(self, config, state=None, distributed=False):
        super().__init__(config)
        self.config = config
        self.ddp = distributed
        self.model = BERTModel(config).to(config.device)
        self.optimizer = self.model.config_optimizer(lr=config.max_lr)
        self.loader = BERTPreTrainLoader(config)
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler('cuda') 
        self.scheduler = None
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
            'mlm_f1': [],
            'nsp_f1': [],
            'perplexity': [],
            'learn_rate': [] 
            }
        
        self.set_scheduler(config.lr_sched)
        self.load_state(state)
        if self.ddp:
            self.init_distributed()

    def print_progress(self, tloss, vloss, mlm_f1, nsp_f1, perp, lr):
        print(f'Step: {Fore.CYAN}{self.iteration}{Fore.RESET}',
            f"| Train Loss: {Fore.CYAN}{tloss:.3f}{Fore.RESET}",
            f"| Eval Loss: {Fore.CYAN}{vloss:.3f}{Fore.RESET}",
            f"| MLM F1: {Fore.CYAN}{mlm_f1:.3f}{Fore.RESET}",
            f"| NSP F1: {Fore.CYAN}{nsp_f1:.3f}{Fore.RESET}",
            f"| Perplex: {Fore.CYAN}{perp:.2f}{Fore.RESET}",
            f"| LR: {Fore.CYAN}{lr:.10f}{Fore.RESET}")
            
    def plot_results(self):
        """Plots results of training session."""
        train_loss = self.results['train_loss']
        eval_loss = self.results['eval_loss']
        mlm_f1 = self.results['mlm_f1']
        nsp_f1 = self.results['nsp_f1']
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

        # Plot MLM / NSP F1
        plt.subplot(2, 2, 2)  
        plt.plot(iters, mlm_f1, label='MLM F1')
        plt.plot(iters, nsp_f1, label='NSP F1')
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

    ### Only using F1 score for logging, but accuracy methods are also available ###
    def get_nsp_accuracy(self, logits, target):
        preds = torch.softmax(logits, dim=-1).argmax(dim=-1)
        return (preds == target).float().mean().item()
    
    def get_mlm_accuracy(self, logits, target):
        preds = torch.softmax(logits, dim=-1).argmax(dim=-1)
        mask = (target != -100)  # Create a mask for targets that are not -100
        # Apply the mask to both predictions and targets
        valid_preds = preds[mask]
        valid_targets = target[mask]
        # Calculate the accuracy only on masked positions
        correct_predictions = valid_preds == valid_targets
        if len(valid_targets) > 0:
            accuracy = correct_predictions.float().mean().item()
        else: # Avoid division by zero if no valid targets
            accuracy = float('nan')  
        return accuracy

    def calculate_nsp_f1(self, logits, target):
        preds = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).flatten()
        target = target.flatten()
        # Calculate F1 score using sklearn for simplicity
        return f1_score(target.cpu().numpy(), preds.cpu().numpy(), average='weighted', zero_division=1)
    
    def calculate_mlm_f1(self, logits, target):
        preds = torch.softmax(logits, dim=-1).argmax(dim=-1)
        mask = (target != -100)  # Mask to identify relevant targets
        # Only consider the positions that are not masked with -100
        valid_preds = preds[mask]
        valid_targets = target[mask]
        # Ensure there are valid targets to evaluate
        if len(valid_targets) > 0:
            # Calculate F1 score using sklearn
            f1 = f1_score(valid_targets.cpu().numpy(), valid_preds.cpu().numpy(), average='weighted', zero_division=1)
        else:  # Avoid division by zero if no valid targets
            f1 = float('nan')  
        return f1
    
    def train_step(self, batch):
        """Train step using Autocast and Gradscaler. Performs N number of gradient
        accumulation steps before stepping the optimizer and scheduler."""

        self.model.train()
        # Mixed precision training with torch.cuda.amp.autocast()
        # Automatically cast dtype during training
        with autocast(device_type=self.config.device, dtype=self.config.dtype): 
            # Forward pass
            mlm_logits, nsp_logits = self.model(batch['input_ids'], batch['segments'], batch['attn_mask'])
            # Calculate mlm and nsp loss
            mlm_loss = self.criterion(mlm_logits.view(-1, mlm_logits.size(-1)), batch['labels'].view(-1))
            nsp_loss = self.criterion(nsp_logits, batch['is_next'])
            loss = mlm_loss + nsp_loss
            # Scale the loss
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
        of grad scale iterations to remain consistent with the training loss output."""

        valid_loss, mlm_f1, nsp_f1, perplex = 0, 0, 0, 0
        self.model.eval()
        with torch.inference_mode():
            for _ in range(self.grad_accum_steps):
                batch = next(self.loader) # Calls get_valid_batch() via __next__
                # Use autocast with validation step to remain consistent in loss calculation
                with autocast(device_type=self.config.device, dtype=self.config.dtype):
                    # Forward pass
                    mlm_logits, nsp_logits = self.model(batch['input_ids'], batch['segments'], batch['attn_mask'])
                    # Calculate mlm and nsp loss
                    mlm_loss = self.criterion(mlm_logits.view(-1, mlm_logits.size(-1)), batch['labels'].view(-1))
                    nsp_loss = self.criterion(nsp_logits, batch['is_next'])
                    loss = mlm_loss + nsp_loss

                if self.config.lr_sched == 'rlrp':
                    self.scheduler.step(loss)
                # Update validation trackers
                valid_loss += loss.item()
                mlm_f1 += self.calculate_mlm_f1(mlm_logits, batch['labels'])
                nsp_f1 += self.calculate_nsp_f1(nsp_logits, batch['is_next'])
                perplex += torch.exp(loss).item()

        return {'loss': valid_loss, 
                'mlm_f1': mlm_f1, 
                'nsp_f1': nsp_f1, 
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
        
        # Accumulate train loss per iteration
        train_loss = 0
        start_time = time.time()
        for i, batch in enumerate(self.loader): # Calls get_train_batch() iteratively via __iter__

            if (i + 1) == self.max_iters:
                print(f"{Fore.MAGENTA}Training complete!{Fore.RESET}")
                self.print_checkpoint_info(saved_dir=False)
                break
            
            self.iteration += 1
            train_loss += self.train_step(batch)

            # Validation step every 100 training iterations
            if self.iteration % 100 == 0:
    
                metrics = self.valid_step()

                # Calculate average loss, accuracy, f1, perplixity and log the results
                train_loss /= 100
                valid_loss = metrics['loss'] / self.grad_accum_steps
                mlm_f1 = metrics['mlm_f1'] / self.grad_accum_steps
                nsp_f1 = metrics['nsp_f1'] / self.grad_accum_steps
                perplex = metrics['perplex'] / self.grad_accum_steps
                
                # Get the current learning rate and print a status update
                current_lr = self.optimizer.param_groups[0]['lr']
                self.print_progress(train_loss, valid_loss, mlm_f1, nsp_f1, perplex, current_lr)
                if self.iteration % 1000 == 0:
                    print(f'Elapsed Training Time: {formatted_time(start_time)}')

                # Update results dictionary for plotting
                self.results['train_loss'].append(train_loss)
                self.results['eval_loss'].append(valid_loss)
                self.results['mlm_f1'].append(mlm_f1)
                self.results['nsp_f1'].append(nsp_f1)
                self.results['perplexity'].append(perplex)
                self.results['learn_rate'].append(current_lr)

                if save_cp:
                    self.save_checkpoint(train_loss, current_lr, best_model)
                else: # Update train loss and LR trackers if not saving best model
                    self.train_loss = train_loss
                    self.current_lr = current_lr

                # Reset train loss tracker
                train_loss = 0

        if plot_results:
            self.plot_results()

    def test(self):
        """Test the model's performance after a training session."""
        self.model.eval()
        with torch.inference_mode():
            batch = self.loader.get_valid_batch()
            # Forward pass
            mlm_logits, nsp_logits = self.model(batch['input_ids'], batch['segments'], batch['attn_mask'])
            # Calculate mlm and nsp loss
            mlm_loss = self.criterion(mlm_logits.view(-1, mlm_logits.size(-1)), batch['labels'].view(-1))
            nsp_loss = self.criterion(nsp_logits, batch['is_next'])

        loss = mlm_loss + nsp_loss
        mlm_f1 = self.get_mlm_accuracy(mlm_logits, batch['labels'])
        nsp_f1 = self.get_nsp_accuracy(nsp_logits, batch['is_next'])
        perplexity = torch.exp(loss)

        print(f"{Fore.GREEN}Model Evaluation:{Fore.RESET}")
        print(f"Test Loss: {Fore.CYAN}{loss:.3f}{Fore.RESET}",
            f"| MLM F1: {Fore.CYAN}{mlm_f1:.3f}{Fore.RESET}",
            f"| NSP F1: {Fore.CYAN}{nsp_f1:.3f}{Fore.RESET}",
            f"| Perplexity: {Fore.CYAN}{perplexity:.2f}{Fore.RESET}")

        # Fill-in the blank test
        prompt = "The [MASK] [MASK] is going [MASK] [MASK] way to the [MASK] [MASK]."
        print(f'{Fore.GREEN}Prompt:{Fore.RESET} {prompt}')
        encoded = self.loader.tokenizer.encode(prompt).ids
        mask_positions = [i for i, token in enumerate(encoded) if token == self.loader.tokenizer.token_to_id('[MASK]')]

        # Predict the masked token positions
        generated_ids = self.model.fill_in(encoded, mask_positions)
        generated_text = self.loader.tokenizer.decode(generated_ids)
        print(f'{Fore.GREEN}Fill in the blank:{Fore.RESET} {generated_text}')


class BERTFineTuner(Trainer):
    """
    Custom trainer class for fine-tuning on large (multi-file) csv file datasets.\n
    Params `config`:\n 
        device: 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype: set to bfloat16 or float16 for autcasting dtype
        ntoken: size of tokenizer vocabulary 
        seqlen: length of token sequence per mini-batch
        nbatch: number of mini-batches (token sequences)
        dmodel: embedding dimension size
        nhead: number of attention heads 
        nlayer: number of encoder/decoder layers
        shead: dmodel // nhead = individual head size
        dropout: % of nodes turned off during training
        bias: optionally turn bias on/off in linear/normalization layers
        nlabels: number of NER labels
        data_dir: path to dataset (single file)
        load_path: path to load model checkpoint/weights
        save_path: path to save model checkpoint/weights 
        lr_sched: LR scheduler type - warmup, cyclic, decay, rlrp, or None
        grad_accum: gradient accumulation steps to simulate larger batch sizes
        max_iters: max training iterations
        wu_iters: warmup iterations for warmup scheduler
        sched_iters: scheduler iterations - adjust for gradient accumulation
        max_lr: max or starting learning rate
        min_lr: minimum learning rate
        step_up: step up iterations for CLR scheduler
        step_dn: step down iterations for CLR scheduler
        epochs: number of epochs for the training dataset
    Params `state`:\n
        None = fine-tune from a pre-trained BERT model\n
        'weights' = load model/optimizer state_dicts to continue fine-tuning on a new dataset\n
    Attributes:\n
        ddp: bool = False | Distributed training.
        model: object = BERTModel()
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
    def __init__(self, config, state=None, distributed=False):
        super().__init__(config)
        self.config = config
        self.ddp = distributed
        self.model = BERTClassifier.from_pretrained(config)
        self.optimizer = self.model.config_optimizer(base_lr=config.base_lr, cls_lr=config.max_lr)
        self.loader = BERTClassifierLoader(config)
        self.ner_criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.sent_criterion = nn.MSELoss()
        self.alpha = 0.5 # Apply weighted loss modifier
        self.scaler = GradScaler('cuda') 
        self.scheduler = None
        self.iteration = 1 # Starting position
        self.max_iters = config.max_iters
        self.grad_accum_steps = config.grad_accum
        self.train_data_path = config.data_dir
        self.saved_data_path = "" # To hold saved dataset directory from last session
        self.train_loss = float('inf')
        self.current_lr = float('inf')
        self.results = {
            'train_loss': [],
            'eval_loss': [],
            'ner_acc': [],
            'ner_f1': [],
            'sent_mse': [],
            'sent_mae': [],
            'learn_rate': [] 
            }
        
        self.set_scheduler(config.lr_sched)
        self.load_state(state)
        if self.ddp:
            self.init_distributed()
        
    def print_progress(self, tloss, vloss, ner_acc, ner_f1, sent_mse, sent_mae, lr):
        print(f'Step: {Fore.CYAN}{self.iteration}{Fore.RESET}',
            f"| Train Loss: {Fore.CYAN}{tloss:.3f}{Fore.RESET}",
            f"| Eval Loss: {Fore.CYAN}{vloss:.3f}{Fore.RESET}",
            f"| NER Acc: {Fore.CYAN}{ner_acc:.3f}{Fore.RESET}",
            f"| NER F1: {Fore.CYAN}{ner_f1:.3f}{Fore.RESET}",
            f"| Sent MSE: {Fore.CYAN}{sent_mse:.3f}{Fore.RESET}",
            f"| Sent MAE: {Fore.CYAN}{sent_mae:.3f}{Fore.RESET}",
            f"| LR: {Fore.CYAN}{lr:.10f}{Fore.RESET}")
        
    def plot_results(self):
        """Plots results of training session."""
        train_loss = self.results['train_loss']
        eval_loss = self.results['eval_loss']
        ner_acc = self.results['ner_acc']
        ner_f1 = self.results['ner_f1']
        sent_mse = self.results['sent_mse']
        sent_mae = self.results['sent_mae']
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

        # Plot NER Accuracy / F1
        plt.subplot(2, 2, 2)  
        plt.plot(iters, ner_acc, c='r', label='NER Acc')
        plt.plot(iters, ner_f1, c='m', label='NER F1')
        plt.title('NER Accuracy / F1')
        plt.xlabel('Iterations x 100')
        plt.ylabel('Acc / F1')
        plt.legend()

        # Plot Senitment MSE / MAE
        plt.subplot(2, 2, 3)  
        plt.plot(iters, sent_mse, c='r', label='Sent MSE')
        plt.plot(iters, sent_mae, c='m', label='Sent MAE')
        plt.title('Sentiment MSE / MAE')
        plt.xlabel('Iterations x 100')
        plt.ylabel('MSE / MAE')
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

    def calculate_rsquared(self, logits, target):
        ss_res = torch.sum((target - logits) ** 2)
        ss_tot = torch.sum((target - torch.mean(target)) ** 2)
        r_squared = 1 - ss_res / ss_tot
        return r_squared.item()

    def calculate_mse(self, logits, targets):
        predictions = torch.softmax(logits, dim=-1).cpu().detach().numpy()
        return mean_squared_error(targets.cpu().numpy(), predictions)
    
    def calculate_mae(self, logits, targets):
        predictions = torch.softmax(logits, dim=-1).cpu().detach().numpy()
        return mean_absolute_error(targets.cpu().numpy(), predictions)
    
    def train_step(self, batch):
        """Train step using Autocast and Gradscaler. Performs N number of gradient
        accumulation steps before stepping the optimizer and scheduler."""

        self.model.train()
        # Mixed precision training with torch.cuda.amp.autocast()
        # Automatically cast dtype during training
        with autocast(device_type=self.config.device, dtype=self.config.dtype): 
            # Forward pass
            ner_logits, sent_logits = self.model(batch['input_ids'], attn_mask=batch['attn_mask'])
            # Calculate NER and sentiment loss
            ner_loss = self.ner_criterion(ner_logits.view(-1, ner_logits.size(-1)), batch['ner_labels'].view(-1))
            sent_loss = self.sent_criterion(F.softmax(sent_logits, dim=-1), batch['sent_labels'])
            total_loss = ner_loss + sent_loss
            # total_loss = self.alpha * ner_loss + (1 - self.alpha) * sent_loss
            # Scale the loss
            scaled_loss = total_loss / self.grad_accum_steps
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

        return total_loss.item()
    
    def valid_step(self):
        """Validation step every 100 iterations. Uses Autocast and does the same number 
        of grad scale iterations to remain consistent with the training loss output."""

        valid_loss, ner_acc, ner_f1, sent_mse, sent_mae = 0, 0, 0, 0, 0
        self.model.eval()
        with torch.inference_mode():
            for _ in range(self.grad_accum_steps):
                batch = next(self.loader) # Calls get_valid_batch() via __next__
                # Use autocast with validation step to remain consistent in loss calculation
                with autocast(device_type=self.config.device, dtype=self.config.dtype):
                    ner_logits, sent_logits = self.model(batch['input_ids'], attn_mask=batch['attn_mask'])
                    # Calculate NER and sentiment loss
                    ner_loss = self.ner_criterion(ner_logits.view(-1, ner_logits.size(-1)), batch['ner_labels'].view(-1))
                    sent_loss = self.sent_criterion(F.softmax(sent_logits, dim=-1), batch['sent_labels'])
                    total_loss = ner_loss + sent_loss
                    # total_loss = self.alpha * ner_loss + (1 - self.alpha) * sent_loss

                if self.config.lr_sched == 'rlrp':
                    self.scheduler.step(total_loss)
                # Update validation trackers
                valid_loss += total_loss.item()
                ner_acc += self.get_accuracy(ner_logits, batch['ner_labels'])
                ner_f1 += self.calculate_f1(ner_logits, batch['ner_labels'])
                sent_mse += self.calculate_mse(sent_logits, batch['sent_labels'])
                sent_mae += self.calculate_mae(sent_logits, batch['sent_labels'])

        return {'loss': valid_loss, 
                'ner_acc': ner_acc, 
                'ner_f1': ner_f1, 
                'sent_mse': sent_mse,
                'sent_mae': sent_mae
                }

    def train(self, save_cp=True, best_model=False, plot_results=True):
        """Model training loop with a validation step every 100 train iterations.\n
        Args:
            save_cp: bool = True | save model and training dataset checkpoint every 100 iterations\n
            best_model: bool = False | only save if it is the best training loss recorded so far\n
            plot_results: bool = True | plot results after training has stopped"""

        print(f"{Fore.MAGENTA}Training Directory for this session:{Fore.RESET} {self.train_data_path}")
        option = input(f'{Fore.YELLOW}Do you want to begin training (y or n)?:{Fore.RESET} ')
        if option == 'y':
            print(f"{Fore.MAGENTA}Begin training session...{Fore.RESET}")
            print(f"{Fore.MAGENTA}Epoch: {Fore.RESET}{Fore.CYAN}0{Fore.RESET}")
        else:
            print(f"{Fore.RED}Abort training...{Fore.RESET}")
            return
        
        # Accumulate train loss per iteration
        train_loss = 0
        start_time = time.time()
        for i, batch in enumerate(self.loader): # Calls get_train_batch() iteratively via __iter__

            if (i + 1) == self.max_iters:
                print(f"{Fore.MAGENTA}Training complete!{Fore.RESET}")
                self.print_checkpoint_info(saved_dir=False)
                break
            
            self.iteration += 1
            train_loss += self.train_step(batch)

            # Validation step every 100 training iterations
            if self.iteration % 100 == 0:
    
                metrics = self.valid_step()

                # Calculate average loss, accuracy, f1, perplixity and log the results
                train_loss /= 100
                valid_loss = metrics['loss'] / self.grad_accum_steps
                ner_acc = metrics['ner_acc'] / self.grad_accum_steps
                ner_f1 = metrics['ner_f1'] / self.grad_accum_steps
                sent_mse = metrics['sent_mse'] / self.grad_accum_steps
                sent_mae = metrics['sent_mae'] / self.grad_accum_steps
                
                # Get the current learning rate and print a status update
                current_lr = self.optimizer.param_groups[2]['lr'] # classifier optim group
                self.print_progress(train_loss, valid_loss, ner_acc, ner_f1, sent_mse, sent_mae, current_lr)
                if self.iteration % 1000 == 0:
                    print(f'Elapsed Training Time: {formatted_time(start_time)}')

                # Update results dictionary for plotting
                self.results['train_loss'].append(train_loss)
                self.results['eval_loss'].append(valid_loss)
                self.results['ner_acc'].append(ner_acc)
                self.results['ner_f1'].append(ner_f1)
                self.results['sent_mse'].append(sent_mse)
                self.results['sent_mae'].append(sent_mae)
                self.results['learn_rate'].append(current_lr)

                if save_cp:
                    self.save_checkpoint(train_loss, current_lr, best_model)
                else: # Update train loss and LR trackers if not saving best model
                    self.train_loss = train_loss
                    self.current_lr = current_lr

                # Reset train loss tracker
                train_loss = 0

        if plot_results:
            self.plot_results()

    def test(self):
        """Test the model's performance after a training session."""
        self.model.eval()
        with torch.inference_mode():
            batch = self.loader.get_valid_batch()
            # Forward pass
            ner_logits, sent_logits = self.model(batch['input_ids'], attn_mask=batch['attn_mask'])
            # Calculate NER and sentiment loss
            ner_loss = self.ner_criterion(ner_logits.view(-1, ner_logits.size(-1)), batch['ner_labels'].view(-1))
            sent_loss = self.sent_criterion(F.softmax(sent_logits, dim=-1), batch['sent_labels'])
            total_loss = ner_loss + sent_loss

        ner_acc = self.get_accuracy(ner_logits, batch['ner_labels'])
        ner_f1 = self.calculate_f1(ner_logits, batch['ner_labels'])
        sent_mse = self.calculate_mse(sent_logits, batch['sent_labels'])
        sent_mae = self.calculate_mae(sent_logits, batch['sent_labels'])

        print(f"{Fore.GREEN}Model Evaluation:{Fore.RESET}")
        print(f"Test Loss: {Fore.CYAN}{total_loss:.3f}{Fore.RESET}",
            f"| NER Acc: {Fore.CYAN}{ner_acc:.3f}{Fore.RESET}",
            f"| NER F1: {Fore.CYAN}{ner_f1:.3f}{Fore.RESET}",
            f"| Sent MSE: {Fore.CYAN}{sent_mse:.3f}{Fore.RESET}",
            f"| Sent MAE: {Fore.CYAN}{sent_mae:.3f}{Fore.RESET}",)

        # Output NER tagging results
        text = "Let's test this sentence with the name John Smith who was born in July and lives in Tampa, Florida."
        self.loader.tokenizer.no_padding()
        encoded = self.loader.tokenizer.encode(text)
        self.model.eval()
        with torch.inference_mode():
            input_tensor = torch.tensor([encoded.ids], dtype=torch.long).to(self.config.device)
            ner_logits, sent_logits = self.model(input_tensor)

        ner_ids = F.softmax(ner_logits, dim=-1).argmax(dim=-1)
        decoded = self.loader.ner_decoder(ner_ids.squeeze(0).tolist(), skip_special=False)
        sentiment = F.softmax(sent_logits, dim=-1).squeeze(0).tolist()

        print(f'{Fore.GREEN}Input Text:{Fore.RESET} {text}')
        print(f'{Fore.GREEN}Input Tokens:{Fore.RESET} {encoded.tokens}')
        print(f'{Fore.GREEN}NER Decoded Tags:{Fore.RESET} {decoded}')
        print(f'{Fore.GREEN}Negative:{Fore.RESET} {sentiment[0]:.3f}',
            f'{Fore.GREEN}Neutral:{Fore.RESET} {sentiment[1]:.3f}',
            f'{Fore.GREEN}Positive:{Fore.RESET} {sentiment[2]:.3f}')
        
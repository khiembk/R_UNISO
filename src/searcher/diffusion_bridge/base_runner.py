import argparse
import datetime
import time

import yaml
import os
import traceback

import torch

from abc import ABC, abstractmethod
from tqdm.autonotebook import tqdm

from searcher.diffusion_bridge.EMA import EMA
from searcher.diffusion_bridge.diff_utils import make_save_dirs, remove_file, sampling_data_from_GP, create_train_dataloader, create_val_dataloader, sampling_from_offline_data, testing_by_oracle, load_metadata_from_task_name, compute_mean_std_tensor, transfor_x2str
import numpy as np
import omegaconf
import gpytorch 
from gaussian_process.GPlib import ExactGPModel
from gaussian_process.GP import GP
import design_bench
from src.models import EncoderDecoderModule
import rootutils
root_dir = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True) 
from src.tasks import get_tasks, get_tasks_from_suites
from src.data.omnipred_datamodule import OmnipredDataModule
#### Ignore the normalize
class BaseRunner(ABC):
    def __init__(self, config, model:EncoderDecoderModule, datamodule:OmnipredDataModule):
        self.net = None  # Neural Network
        self.optimizer = None  # optimizer
        self.scheduler = None  # scheduler
        self.config = config  # config from configuration file
        # set training params
        self.global_epoch = 0  # global epoch
        self.global_step = 0
        # set datamodule
        self.datamodule = datamodule
        # set encoder-decoder
        self.encoder_model = model.encoder
        self.decoder_model = model.rec_model  
        self.input_tokenizer = model.input_tokenizer
        self.shared = model.shared
        self._emb_metadata = model._emb_metadata
        self.frozen_encoder_decoder()
        # orginal code
        self.GAN_buffer = {}  # GAN buffer for Generative Adversarial Network
        self.topk_checkpoints = {}  # Top K checkpoints

        # set log and save destination
        
        self.result = argparse.Namespace()
        if self.config.args.train:
            self.result.ckpt_path = make_save_dirs(self.config.args,
                                                    prefix=self.config.task.name + f'/seed{self.config.args.seed}',
                                                    suffix=self.config.model.model_name,
                                                    with_time=False)

            #self.save_config()  # save configuration file
            
        # initialize model
        self.net, self.optimizer, self.scheduler = self.initialize_model_optimizer_scheduler(self.config)
        

        # initialize EMA
        self.use_ema = False if not self.config.model.__contains__('EMA') else self.config.model.EMA.use_ema
        if self.use_ema:
            self.ema = EMA(self.config.model.EMA.ema_decay)
            self.update_ema_interval = self.config.model.EMA.update_ema_interval
            self.start_ema_step = self.config.model.EMA.start_ema_step
            self.ema.register(self.net)

        # load model from checkpoint
        self.load_model_from_checkpoint()

        # initialize DDP
        self.net = self.net.to("cuda")

        # get offline data from design-bench
        
        #self.offline_x, self.offline_z, self.offline_y, self.metadata = self.get_offline_feature_z()
        self.get_data_label_xy()
        ### 
        self.distinct_meta = self.get_distinct_meta()
        self.distinct_task_name = self.get_distinct_task_name()
        # if self.config.task.normalize_z:
        #     self.offline_z = (self.offline_z - self.mean_offline_z) / self.std_offline_z
        # if self.config.task.normalize_y:
        #     self.offline_y = (self.offline_y - self.mean_offline_y) / self.std_offline_y
    
        # self.offline_z = self.offline_z.to(self.config.training.device[0])
        # self.offline_y = self.offline_y.to(self.config.training.device[0])
    
    def frozen_encoder_decoder(self):
        ### frozen encoder-decoder
        for p in self.decoder_model.parameters():
            p.requires_grad = False
        for p in self.encoder_model.parameters():
            p.requires_grad = False
        for p in self.shared.parameters():
            p.requires_grad = False
        
    
    def get_distinct_task_name(self):
        seen = []
        for task_name in self.task_names:
        # Check if meta is already in seen using torch.equal
            if not any((task_name[0] == s) for s in seen):
               seen.append(task_name[0])
        #print("task_names: ", seen)
        return seen

    def generate_data_with_GP(self, metadata, lengthscale, variance, noise, mean_prior, epoch=0, num_points = 10):
        self.offline_x_m, self.offline_y_m = self.load_xy_by_metadata(metadata)
        #print("offline y: ",self.offline_y_m)    
        #print("offline x: ",self.offline_x_m)          
        if self.config.GP.type_of_initial_points == 'highest':
            best_indices = torch.argsort(torch.tensor(self.offline_y_m))[ -num_points:]
            # print("best indicate:", best_indices)
            self.best_x_m = [self.offline_x_m[i] for i in best_indices]
        else: 
            best_indices = torch.argsort(torch.tensor(self.offline_y_m))[: num_points]
            # print("best indicate:", best_indices)            
            self.best_x_m = [self.offline_x_m[i] for i in best_indices]
        
        #### init data
        x_m_2d = torch.stack(self.offline_x_m)
        x_m_2d = x_m_2d.to(dtype=torch.float32, device="cuda")
        x_m_2d_best = torch.stack(self.best_x_m)
        x_m_2d_best = x_m_2d_best.to(dtype=torch.float32, device="cuda")
                    ### init GP model
        GP_Model = GP(device= "cuda",
                                x_train= x_m_2d,
                                y_train= torch.tensor(self.offline_y_m).to(device="cuda"), 
                                lengthscale=lengthscale, 
                                variance=variance, 
                                noise=noise, 
                                mean_prior=mean_prior)
                    ### generate data from GP
        data_from_GP = sampling_data_from_GP(x_train= x_m_2d_best,
                                                    device= "cuda",
                                                    GP_Model=GP_Model,
                                                    num_functions=self.config.GP.num_functions,
                                                    num_gradient_steps=self.config.GP.num_gradient_steps,
                                                    num_points= num_points,
                                                    learning_rate=self.config.GP.sampling_from_GP_lr,
                                                    delta_lengthscale=self.config.GP.delta_lengthscale,
                                                    delta_variance=self.config.GP.delta_variance,
                                                    seed=epoch,
                                                    threshold_diff=self.config.GP.threshold_diff)
        
        return data_from_GP



    @torch.no_grad()
    def get_data_label_xy(self):
        print("Begin: load the data(x,y)...")
        #### get initial train loader 
        train_dataloader = self.datamodule.train_dataloader()
        pbar = tqdm(train_dataloader, total=len(train_dataloader), smoothing=0.01, disable=False)
        x_offline = []
        y_offline = []
        task_names = []
        batch_count = 0
        max_batches = 50
        meta_offline = []
        task_to_metadata = {}
        for batch in pbar:
            with torch.no_grad():
                task_names.append(batch["task_name"])
                y_offline.append(batch["ori_y"])
                m_embeddings = self._emb_metadata(batch["metadata"])
                meta_offline.append(m_embeddings)
                x_offline.append(batch["ori_x"])

                for i, task_name in enumerate(batch["task_name"]):
                    if task_name not in task_to_metadata:
                       task_to_metadata[task_name] = m_embeddings[i]
     
            batch_count += 1
            if batch_count >= max_batches:
                pbar.close()  # Properly close the progress bar
                break

        #print("dic name_task-metadata: ", task_to_metadata)    
        self.task_names = task_names
        self.x_offline = x_offline
        self.y_offline = y_offline
        self.metadata =  meta_offline
        self.task_to_metadata = task_to_metadata
      
    @torch.no_grad()
    def get_offline_feature_z(self):
        # frozen encoder-decoder
        self.frozen_encoder_decoder()
        print("Begin: extract feature...")
        #### get initial train loader 
        train_dataloader = self.datamodule.train_dataloader()
        pbar = tqdm(train_dataloader, total=len(train_dataloader), smoothing=0.01, disable=False)
        batch_count = 0
        max_batches = 50
       
        z_offline = []
    
        for batch in pbar:
            with torch.no_grad():
                input_embeds = self.shared(batch["input_ids"])
                encoder_outputs = self.encoder_model(
                inputs_embeds=input_embeds, attention_mask= batch["attention_mask"])
                encoder_hidden_states = encoder_outputs.last_hidden_state
                mean_pooled = self._mean_pooling(encoder_hidden_states, batch["attention_mask"])
                z_offline.append(mean_pooled)
            batch_count += 1
            if batch_count >= max_batches:
                pbar.close()  # Properly close the progress bar
                break
        return  z_offline
    
    @torch.no_grad()
    def transform_x_2token(self, x_np, name_task):
        x_str = transfor_x2str(x_np, name_task)
        x_tokens = self.input_tokenizer(x_str, return_tensors="pt",  
        padding=True, truncation=True)

        return {
            "input_ids": x_tokens["input_ids"].squeeze(),
            "attention_mask": x_tokens["attention_mask"].squeeze(),
            "name_task": name_task 
            }

    @torch.no_grad()
    def transform_x_token_2z(self, x_tokens):
        input_ids = x_tokens["input_ids"]
        attention_mask = x_tokens["attention_mask"]
        # transform to [batch_size, _]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  
            attention_mask = attention_mask.unsqueeze(0)
        input_embeds = self.shared(input_ids)
        encoder_outputs = self.encoder_model(
        inputs_embeds=input_embeds, attention_mask= attention_mask)
        encoder_hidden_states = encoder_outputs.last_hidden_state
        mean_pooled = self._mean_pooling(encoder_hidden_states, attention_mask)
               
        return mean_pooled

    @torch.no_grad()
    def get_offline_feature_z_from_GP(self, datasets, name_task):
        # frozen encoder-decoder
        self.frozen_encoder_decoder()
        print("Begin: extract feature from GP")
        #### get initial train loader 
        z_datasets= []
        for key in datasets:
            # z_offline[key] = []
            
            samples = datasets[key]

            for sample in tqdm(samples, desc=f"Processing samples for {key}"):
                (high_x, high_y), (low_x, low_y) = sample
                z_high = self.transform_x2_z(high_x, name_task)
                z_low = self.transform_x2_z(low_x, name_task)
                #print("size of z: ",z_low.shape)
                z_sample = [(z_high.detach(),high_y),(z_low.detach(),low_y), name_task]
                z_datasets.append(z_sample)
        
        return z_datasets

    
    @torch.no_grad()
    def transform_x2_z(self, x, task_name):
        x_tokens = self.transform_x_2token(x, task_name)
        return self.transform_x_token_2z(x_tokens)


    def load_feature_by_metadata(self, metadata_val):
        z_offline= []
        y_offline = []
        for i in range(len(self.offline_z)):
            if torch.equal(self.metadata[i], metadata_val): 
                z_offline.append(self.offline_z[i])
                y_offline.append(self.offline_y[i])

        return z_offline, y_offline

    def load_xy_by_metadata(self, metadata_val):
        x_offline= []
        y_offline = []
        for i in range(len(self.x_offline)):
            # print("metadata: ", self.metadata[i][0])
            # print("meta_val: ", metadata_val)
            # pass
            if torch.equal(self.metadata[i][0], metadata_val): 
                x_offline.append(self.x_offline[i][0])
                y_offline.append(float(self.y_offline[i][0]))

        return x_offline, y_offline
        
    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


    @torch.no_grad()
    def _decode_x(self, z: torch.Tensor, task) -> np.ndarray:
        if z.dim() == 3 and z.size(1) == 1:
            z = z.squeeze(1)
        print("shape of z: ", z.shape)
        z.to("cuda")
        self.decoder_model.to("cuda")
        ### decoder form z to x
        x_res = self.decoder_model(z)
        x_res = x_res.clone()
        input_ids = torch.argmax(x_res, dim=-1)
        #print("input_ids: ", input_ids[0])

        vocab_size = self.input_tokenizer.vocab_size
        if input_ids.min().item() < 0 or input_ids.max().item() >= vocab_size:
            print(f"Warning: input_ids contains invalid values (min: {input_ids.min().item()}, max: {input_ids.max().item()}, vocab_size: {vocab_size})")
            input_ids = torch.clamp(input_ids, min=0, max=vocab_size - 1)
    
   
        #print(f"input_ids shape: {input_ids.shape}, dtype: {input_ids.dtype}, device: {input_ids.device}")
    
    
        x_str_list = []
        for ids in input_ids:
            try:
               x_str = self.input_tokenizer.decode(ids, skip_special_tokens=True).strip()
               x_str_list.append(x_str)
            except Exception as e:
               print(f"Error decoding input_ids {ids}: {e}")
               x_str_list.append("")  # Fallback for failed decoding
        
        print("x_str: ", x_str_list[0])
        return x_str_list
            
      

    
    # print msg
    def logger(self, msg, **kwargs):
        print(msg, **kwargs)

    # save configuration file
    def save_config(self):
        save_path = os.path.join(self.result.ckpt_path, 'config.yaml')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_config = self.config
        with open(save_path, 'w') as f:
            yaml.dump(save_config, f)

    def initialize_model_optimizer_scheduler(self, config, is_test=False):
        """
        get model, optimizer, scheduler
        :param args: args
        :param config: config
        :param is_test: is_test
        :return: net: Neural Network, nn.Module;
                 optimizer: a list of optimizers;
                 scheduler: a list of schedulers or None;
        """
        net = self.initialize_model(config)
        optimizer, scheduler = None, None
        if not is_test:
            optimizer, scheduler = self.initialize_optimizer_scheduler(net, config)
        return net, optimizer, scheduler

    # load model, EMA, optimizer, scheduler from checkpoint
    def load_model_from_checkpoint(self):
        model_states = None
        if self.config.model.__contains__('model_load_path') and self.config.model.model_load_path is not None:
            model_states = torch.load(self.config.model.model_load_path, map_location='cpu')

            self.global_epoch = model_states['epoch']
            self.global_step = model_states['step']

            # load model
            self.net.load_state_dict(model_states['model'])

            # load ema
            if self.use_ema:
                self.ema.shadow = model_states['ema']
                self.ema.reset_device(self.net)

            # load optimizer and scheduler
            if self.config.args.train:
                if self.config.model.__contains__('optim_sche_load_path') and self.config.model.optim_sche_load_path is not None:
                    optimizer_scheduler_states = torch.load(self.config.model.optim_sche_load_path, map_location='cpu')
                    for i in range(len(self.optimizer)):
                        self.optimizer[i].load_state_dict(optimizer_scheduler_states['optimizer'][i])

                    if self.scheduler is not None:
                        for i in range(len(self.optimizer)):
                            self.scheduler[i].load_state_dict(optimizer_scheduler_states['scheduler'][i])
        return model_states

    def get_checkpoint_states(self, stage='epoch_end'):
        optimizer_state = []
        for i in range(len(self.optimizer)):
            optimizer_state.append(self.optimizer[i].state_dict())

        scheduler_state = []
        for i in range(len(self.scheduler)):
            scheduler_state.append(self.scheduler[i].state_dict())

        optimizer_scheduler_states = {
            'optimizer': optimizer_state,
            'scheduler': scheduler_state
        }

        model_states = {
            'step': self.global_step,
        }

        model_states['model'] = self.net.state_dict()

        if stage == 'exception':
            model_states['epoch'] = self.global_epoch
        else:
            model_states['epoch'] = self.global_epoch + 1

        if self.use_ema:
            model_states['ema'] = self.ema.shadow
        return model_states, optimizer_scheduler_states

    # EMA part
    def step_ema(self):
        with_decay = False if self.global_step < self.start_ema_step else True
        self.ema.update(self.net, with_decay=with_decay)

    def apply_ema(self):
        if self.use_ema:
            self.ema.apply_shadow(self.net)

    def restore_ema(self):
        if self.use_ema:
            self.ema.restore(self.net)

    # Evaluation and sample part
    @torch.no_grad()
    def validation_step(self, val_batch, epoch, step):
        self.apply_ema()
        self.net.eval()
        loss = self.loss_fn(net=self.net,
                            batch=val_batch,
                            epoch=epoch,
                            step=step,
                            opt_idx=0,
                            stage='val_step')
        if len(self.optimizer) > 1:
            loss = self.loss_fn(net=self.net,
                                batch=val_batch,
                                epoch=epoch,
                                step=step,
                                opt_idx=1,
                                stage='val_step')
        self.restore_ema()

    @torch.no_grad()
    def validation_epoch(self, val_loader, epoch):
        self.apply_ema()
        self.net.eval()

        pbar = tqdm(val_loader, total=len(val_loader), smoothing=0.01, disable=False)
        step = 0
        loss_sum = 0.
        dloss_sum = 0.
        for val_batch in pbar:
            loss = self.loss_fn(net=self.net,
                                batch=val_batch,
                                epoch=epoch,
                                step=step,
                                opt_idx=0,
                                stage='val',
                                write=False)
            loss_sum += loss
            if len(self.optimizer) > 1:
                loss = self.loss_fn(net=self.net,
                                    batch=val_batch,
                                    epoch=epoch,
                                    step=step,
                                    opt_idx=1,
                                    stage='val',
                                    write=False)
                dloss_sum += loss
            step += 1
        average_loss = loss_sum / step
        self.restore_ema()
        return average_loss


    # abstract methods
    @abstractmethod
    def print_model_summary(self, net):
        pass

    @abstractmethod
    def initialize_model(self, config):
        """
        initialize model
        :param config: config
        :return: nn.Module
        """
        pass

    @abstractmethod
    def initialize_optimizer_scheduler(self, net, config):
        """
        initialize optimizer and scheduler
        :param net: nn.Module
        :param config: config
        :return: a list of optimizers; a list of schedulers
        """
        pass

    @abstractmethod
    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage='train', write=False):
        """
        loss function
        :param net: nn.Module
        :param batch: batch
        :param epoch: global epoch
        :param step: global step
        :param opt_idx: optimizer index, default is 0; set it to 1 for GAN discriminator
        :param stage: train, val, test
        :param write: write loss information to SummaryWriter
        :return: a scalar of loss
        """
        pass

    @abstractmethod
    def sample(self, net, low_candidates, low_scores, high_cond_scores):
        """
        sample a single batch
        :param net: nn.Module
        :param batch: batch
        :param sample_path: path to save samples
        :param stage: train, val, test
        :return:
        """
        pass


    def on_save_checkpoint(self, net, train_loader, val_loader, epoch, step):
        """
        additional operations whilst saving checkpoint
        :param net: nn.Module
        :param train_loader: train data loader
        :param val_loader: val data loader
        :param epoch: epoch
        :param step: step
        :return:
        """
        pass
    def get_distinct_meta(self):
        """
    Get unique metadata values from self.metadata.
    
    Returns:
        list: List of unique metadata tensors.
        """
        seen = []
        for meta in self.metadata:
        # Check if meta is already in seen using torch.equal
            if not any(torch.equal(meta, s) for s in seen):
               seen.append(meta)
        
        return seen
    
    @torch.no_grad()
    def create_feature_loader(self, lengthscale, variance, noise, mean_prior):
        
        feature_datasets = []
        for task_name in self.distinct_task_name:
            metadata = self.task_to_metadata[task_name]
            data_from_GP = self.generate_data_with_GP(metadata= metadata, lengthscale = lengthscale, variance = variance, noise = noise, mean_prior = mean_prior)
            z_datasets = self.get_offline_feature_z_from_GP(data_from_GP, task_name)
            feature_datasets.extend(z_datasets)
                    
                
                    
        train_loader, current_epoch_val_dataset = create_train_dataloader(data_from_GP= feature_datasets,
                                                        val_frac=self.config.training.val_frac,
                                                        batch_size=self.config.training.batch_size,
                                                        shuffle=True)

        return train_dataloader, current_epoch_val_dataset
  


    def train(self):
        
            start_epoch = self.global_epoch
        
        # try:
            # initialize params for GP
            lengthscale = torch.tensor(self.config.GP.initial_lengthscale, device="cuda")
            variance = torch.tensor(self.config.GP.initial_outputscale, device="cuda")
            noise = torch.tensor(self.config.GP.noise, device="cuda")
            mean_prior = torch.tensor(0.0, device = "cuda") 
            
            val_loader = None
            val_dataset = []
            
            print("create dataset...")
            #train_loader, current_epoch_val_dataset = self.create_feature_loader(lengthscale = lengthscale, variance = variance, noise = noise, mean_prior = mean_prior)

            accumulate_grad_batches = self.config.training.accumulate_grad_batches 
            for epoch in range(start_epoch, self.config.training.n_epochs):
                print("Start at ep: ", epoch)
                start_time = time.time()
                
                feature_datasets = []
                for task_name in self.distinct_task_name:
                    metadata = self.task_to_metadata[task_name]
                    data_from_GP = self.generate_data_with_GP(metadata= metadata, lengthscale = lengthscale, variance = variance, noise = noise, mean_prior = mean_prior)
                    z_datasets = self.get_offline_feature_z_from_GP(data_from_GP, task_name)
                    feature_datasets.extend(z_datasets)    

                train_loader, current_epoch_val_dataset = create_train_dataloader(data_from_GP= feature_datasets,
                                                        val_frac=self.config.training.val_frac,
                                                        batch_size=self.config.training.batch_size,
                                                        shuffle=True)

                val_dataset = val_dataset + current_epoch_val_dataset
                
                pbar = tqdm(train_loader, total=len(train_loader), smoothing=0.01, disable=False)
                self.global_epoch = epoch
                for train_batch in pbar:
                       self.global_step += 1
                       self.net.train()

                       losses = []
                       for i in range(len(self.optimizer)):
                            loss = self.loss_fn(net=self.net,
                                            batch=train_batch,
                                            epoch=epoch,
                                            step=self.global_step,
                                            opt_idx=i,
                                            stage='train')

                            loss.backward()
                            if self.global_step % accumulate_grad_batches == 0:
                                self.optimizer[i].step()
                                self.optimizer[i].zero_grad()
                                if self.scheduler is not None:
                                     self.scheduler[i].step(loss)

                            losses.append(loss.detach().mean())

                            if self.use_ema and self.global_step % (self.update_ema_interval*accumulate_grad_batches) == 0:
                                self.step_ema()

                            if len(self.optimizer) > 1:
                                pbar.set_description(
                                (
                                f'Epoch: [{epoch + 1} / {self.config.training.n_epochs}] '
                                f'iter: {self.global_step} loss-1: {losses[0]:.4f} loss-2: {losses[1]:.4f}'
                                ))
                            else:
                                pbar.set_description(
                            (
                                f'Epoch: [{epoch + 1} / {self.config.training.n_epochs}] '
                                f'iter: {self.global_step} loss: {losses[0]:.4f}'
                            ))


                end_time = time.time()
                elapsed_rounded = int(round((end_time-start_time)))
                self.logger("training time: " + str(datetime.timedelta(seconds=elapsed_rounded)))
                
                    # validation
                if (epoch + 1) % self.config.training.validation_interval == 0 or (
                        epoch + 1) == self.config.training.n_epochs:
                        with torch.no_grad():
                            val_loader = create_val_dataloader(val_dataset=val_dataset,
                                                            batch_size=self.config.training.batch_size,
                                                            shuffle=False)
                        
                            average_loss = self.validation_epoch(val_loader, epoch)
                        

                    # save checkpoint
                if (epoch + 1) % self.config.training.save_interval == 0 or \
                        (epoch + 1) == self.config.training.n_epochs:
                        with torch.no_grad():
                            
                            self.on_save_checkpoint(self.net, train_loader, val_loader, epoch, self.global_step)
                            model_states, optimizer_scheduler_states = self.get_checkpoint_states(stage='epoch_end')
                            
                            # save top_k checkpoints
                            model_ckpt_name = f'top_model_epoch_{epoch + 1}.pth'
                            optim_sche_ckpt_name = f'top_optim_sche_epoch_{epoch + 1}.pth'

                            if self.config.args.save_top and (epoch + 1) == self.config.training.n_epochs :
                                print("save top model start...")
                                # wandb.log('save top model start...')
                                top_key = 'top'
                                if top_key not in self.topk_checkpoints:
                                    print('top key not in topk_checkpoints')
                                    # wandb.log('top key not in topk_checkpoints')
                                    self.topk_checkpoints[top_key] = {"loss": average_loss,
                                                                    'model_ckpt_name': model_ckpt_name,
                                                                    'optim_sche_ckpt_name': optim_sche_ckpt_name}

                                    print(f"saving top checkpoint: average_loss={average_loss} epoch={epoch + 1}")
                                    torch.save(model_states,
                                            os.path.join(self.result.ckpt_path, model_ckpt_name))
                                    torch.save(optimizer_scheduler_states,
                                            os.path.join(self.result.ckpt_path, optim_sche_ckpt_name))
                                else:
                                    if average_loss < self.topk_checkpoints[top_key]["loss"]:
                                        print("remove " + self.topk_checkpoints[top_key]["model_ckpt_name"])
                                        remove_file(os.path.join(self.result.ckpt_path,
                                                                self.topk_checkpoints[top_key]['model_ckpt_name']))
                                        remove_file(os.path.join(self.result.ckpt_path,
                                                                self.topk_checkpoints[top_key]['optim_sche_ckpt_name']))

                                        print(
                                        f"saving top checkpoint: average_loss={average_loss} epoch={epoch + 1}")

                                        self.topk_checkpoints[top_key] = {"loss": average_loss,
                                                                        'model_ckpt_name': model_ckpt_name,
                                                                        'optim_sche_ckpt_name': optim_sche_ckpt_name}

                                        torch.save(model_states,
                                                os.path.join(self.result.ckpt_path, model_ckpt_name))
                                        torch.save(optimizer_scheduler_states,
                                                os.path.join(self.result.ckpt_path, optim_sche_ckpt_name))
                                if epoch + 1 == self.config.training.n_epochs:
                                    return os.path.join(self.result.ckpt_path, self.topk_checkpoints[top_key]['model_ckpt_name']), os.path.join(self.result.ckpt_path, self.topk_checkpoints[top_key]['optim_sche_ckpt_name'])
                                                
        # except BaseException as e:
        #     print('str(Exception):\t', str(Exception))
        #     print('str(e):\t\t', str(e))
        #     print('repr(e):\t', repr(e))
        #     print('traceback.print_exc():')
        #     traceback.print_exc()
        #     print('traceback.format_exc():\n%s' % traceback.format_exc())
    
    @torch.no_grad()
    def test(self, task):
        metadata = load_metadata_from_task_name(task)
        m_embeddings = self._emb_metadata(metadata)
        self.offline_z_m, self.offline_y_m = self.load_feature_by_metadata(metadata_val= m_embeddings)
        
        low_candidates, low_scores = sampling_from_offline_data(x=self.offline_z_m,
                                                                y=self.offline_y_m,
                                                                n_candidates=self.config.testing.num_candidates, 
                                                                type=self.config.testing.type_sampling,
                                                                percentile_sampling=self.config.testing.percentile_sampling,
                                                                seed=self.config.args.seed)
        if self.use_ema:
            self.apply_ema()
        self.net.eval()
        
        task_to_min = {'TFBind8-Exact-v0': 0.0, 'TFBind10-Exact-v0': -1.8585268, 'AntMorphology-Exact-v0': -386.90036, 'DKittyMorphology-Exact-v0': -880.4585}
        task_to_max = {'TFBind8-Exact-v0': 1.0, 'TFBind10-Exact-v0': 2.1287067, 'AntMorphology-Exact-v0': 590.24445, 'DKittyMorphology-Exact-v0': 340.90985}
        task_to_best = {'TFBind8-Exact-v0': 0.43929616, 'TFBind10-Exact-v0': 0.005328223, 'AntMorphology-Exact-v0': 165.32648, 'DKittyMorphology-Exact-v0': 199.36252}
        
        # oracle_y_min = task_to_min[self.config.task.name]
        # oracle_y_max = task_to_max[self.config.task.name] 
        # # normalize oracle_y_max by mean and std of offline data
        # normalized_oracle_y_max = (oracle_y_max - self.mean_offline_y) / self.std_offline_y
        # high_cond_scores = torch.full(low_scores.shape, normalized_oracle_y_max.item()*self.config.testing.alpha)
        
        # high_candidates_z = self.sample(self.net, low_candidates, low_scores, high_cond_scores)
        # ### decode the to x space
        # high_candidates = self._decode_x(high_candidates_z)
        # # denormalize high_candidates
        # high_candidates = high_candidates.cpu()
        # denormalize_high_candidates = high_candidates * self.std_offline_x + self.mean_offline_x

        # if task.is_discrete: 
        #     denormalize_high_candidates = denormalize_high_candidates.reshape(denormalize_high_candidates.shape[0],task.x.shape[1],task.x.shape[2])
        
        # high_true_scores = task.predict(denormalize_high_candidates.numpy())
        # # import pdb ; pdb.set_trace()
        # final_score = (torch.from_numpy(high_true_scores) - oracle_y_min)/(oracle_y_max - oracle_y_min)    
        # percentiles = torch.quantile(final_score, torch.tensor([1.0, 0.8, 0.5]), interpolation='higher') 
        
        # return percentiles[0].item(), percentiles[1].item(), percentiles[2].item()
    @torch.no_grad()
    def infer_feature_with_metadata(self, task_name, xs_np ):
        
        z_list = []
        pbar = tqdm(xs_np, total=len(xs_np), smoothing=0.01, disable=False)
        cur_batch = 0
        
        for x_np in pbar:
            z = self.transform_x2_z(x_np, task_name)
            z_list.append(z)
            
        return z_list

    @torch.no_grad()
    def run(self, task_instance, task_name, metadata_string):
        self.offline_x_m = torch.tensor(task_instance.x_np)  # Convert x_np to tensor
        self.offline_y_m = torch.tensor(task_instance.y_np)
        # print("offline y_m: ", self.offline_y_m[0])
        # print("offline x_m: ", self.offline_x_m[0])
        print("len of x_m: ", len(self.offline_x_m))
        print("len of y_m: ", len(self.offline_y_m))
        ### infer feature z
        # print("infer feature z...") 
        # self.offline_z_m = self.infer_feature_with_metadata(task_name= task_name,task_instance = task_instance)
        
        mean_offline_y = np.mean(task_instance.y_np)
        std_offline_y = np.std(task_instance.y_np)
        # x_m_2d = torch.stack(self.offline_x_m)
        # x_m_2d = x_m_2d.to(dtype=torch.float32, device="cuda")
       
        low_candidates_x, low_scores = sampling_from_offline_data(x= self.offline_x_m,
                                                                y= self.offline_y_m,
                                                                n_candidates=self.config.testing.num_candidates, 
                                                                type=self.config.testing.type_sampling,
                                                                percentile_sampling=self.config.testing.percentile_sampling,
                                                                seed=self.config.args.seed)
        if self.use_ema:
            self.apply_ema()
        self.net.eval()
        
        low_candidates_z = self.infer_feature_with_metadata(task_name= task_name, xs_np = low_candidates_x)
        
        low_candidates_z = torch.stack(low_candidates_z)
        
        task_to_min = {'TFBind8-Exact-v0': 0.0, 'TFBind10-Exact-v0': -1.8585268, 'AntMorphology-Exact-v0': -386.90036, 'DKittyMorphology-Exact-v0': -880.4585}
        task_to_max = {'TFBind8-Exact-v0': 1.0, 'TFBind10-Exact-v0': 2.1287067, 'AntMorphology-Exact-v0': 590.24445, 'DKittyMorphology-Exact-v0': 340.90985}
        task_to_best = {'TFBind8-Exact-v0': 0.43929616, 'TFBind10-Exact-v0': 0.005328223, 'AntMorphology-Exact-v0': 165.32648, 'DKittyMorphology-Exact-v0': 199.36252}
        
        oracle_y_min = task_to_min[task_name]
        oracle_y_max = task_to_max[task_name] 
        # normalize oracle_y_max by mean and std of offline data
        normalized_oracle_y_max = (oracle_y_max - mean_offline_y) / std_offline_y
        high_cond_scores = torch.full(low_scores.shape, normalized_oracle_y_max.item()*self.config.testing.alpha)
        
        high_candidates_z = self.sample(self.net, low_candidates_z, low_scores, high_cond_scores)
        ### decode the to x space
        high_candidates = self._decode_x(high_candidates_z, task_instance)
         
        
        return high_candidates
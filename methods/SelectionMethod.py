import torch
import models
import os
import shutil
import time
from dataclasses import dataclass
import random
import numpy as np
from .method_utils import *
import data
from torch.utils.data import DataLoader
# from ema_pytorch import EMA

@dataclass
class MinibatchInfo:
    """Object returned by SelectionMethod.before_batch"""

    inputs: torch.Tensor
    targets: torch.Tensor
    indices: torch.Tensor
    weights: torch.Tensor | None = None  # optional. For weighted loss

class SelectionMethod(object):
    method_name = 'SelectionMethod'
    def __init__(self, config, logger):
        logger.info(f'Creating {self.method_name}...')
        self.config = config
        self.logger = logger
        # create model
        model_type = config['networks']['type']
        model_args = config['networks']['params'] | config['dataset']
        self.model = getattr(models, model_type)(**model_args)
        self.training_opt = config['training_opt']
        self.start_epoch = 0
        self.best_acc = 0
        self.best_epoch = 0
        self.is_best = False
        self.total_time = 0
        self.time_this_epoch = 0
        self.total_step = 0
        self.resume_checkpoint = None
        # gpu
        self.num_gpus = config['num_gpus']
        if self.num_gpus == 0:
            self.model = self.model.cpu()
        elif self.num_gpus == 1:
            self.model = self.model.cuda()
        elif self.num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model).cuda()
        else:
            raise ValueError(f'Wrong number of GPUs: {self.num_gpus}')
        
        # create optimizer
        self.optimizer = create_optimizer(self.model, config)
        self.scheduler = create_scheduler(self.optimizer, config)
        # resume
        config['training_opt']['resume'] = config['training_opt']['resume'] if 'resume' in config['training_opt'] else None
        if config['training_opt']['resume'] is not None:
            self.resume(config['training_opt']['resume'])
        
        # create EMA model
        # self.ema_net = EMA(
        #     self.model,
        #     beta=0.99,
        #     update_after_step=0,
        #     update_every=5,
        # )
        # self.ema_net.eval()

        self.epochs = config['training_opt']['num_epochs'] if 'num_epochs' in config['training_opt'] else None
        self.num_steps = config['training_opt']['num_steps'] if 'num_steps' in config['training_opt'] else None
        if self.epochs is None and self.num_steps is None:
            raise ValueError('Must specify either num_epochs or num_steps in training_opt')
        self.num_data_workers = config['training_opt']['num_data_workers']
        self.batch_size = config['training_opt']['batch_size']

        # data
        self.data_info = getattr(data, config['dataset']['name'])(config, logger)
        self.num_classes = self.data_info['num_classes']
        
        self.train_dset = self.data_info['train_dset']
        self.test_loader = self.data_info['test_loader']
        self.auxiliary_test_loaders = self.data_info.get('auxiliary_test_loaders', []) # TODO: implement in data_info
        self.num_train_samples = self.data_info['num_train_samples']

        self.criterion = create_criterion(config, logger)
        self.need_features = False
        self.train_loader = DataLoader(self.train_dset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_data_workers, pin_memory=True, drop_last=False)
        self.fixed_train_loader = DataLoader(self.train_dset, batch_size=512, shuffle=False, num_workers=self.num_data_workers, pin_memory=True, drop_last=False)
        model_name = self.config['networks']['params'].get('m_type', self.config['networks']['type'])
        dataset_name = self.config['dataset']['name']

        # Diagnostics
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        diagnostics_config = {
            'layers': self.training_opt.get('diagnostics_layers', []),
            'wandb_histogram_max_points': self.training_opt.get('wandb_histogram_max_points', 200000),
            'linear_probe_max_iter': self.training_opt.get('diagnostics_lr_max_iter', 300),
            'wandb_ntk': self.training_opt.get('diagnostics_ntk_enabled', False),
            'ntk_max_samples': self.training_opt.get('diagnostics_ntk_max_samples', 1000),
            'ntk_top_k': self.training_opt.get('diagnostics_ntk_top_k', 10),
            'local_spectral_decay': self.training_opt.get('local_spectral_decay', False),
            'teacher_model_config': self.config,
            **config.get('diagnostics', {}),
        }
        diagnostics_context = DiagnosticsRunContext(
            save_dir=self.config['save_dir'],
            project_root=project_root,
            artifact_stem=self.config['artifact_stem'],
            dataset_name=dataset_name,
            model_name=model_name,
            seed=self.config['seed'],
            fixed_train_loader=self.fixed_train_loader,
            test_loader=self.test_loader,
            total_batches=len(self.train_loader),
            num_train_samples=self.num_train_samples,
            num_epochs=self.epochs,
            num_steps=self.num_steps,
            initial_best_acc=self.best_acc,
            initial_best_epoch=self.best_epoch,
            checkpoint_saver=self.save_checkpoint,
            noisy_indices=self.data_info.get('noisy_indices'),
            true_labels=self.data_info.get('true_labels'),
            auxiliary_test_loaders=self.auxiliary_test_loaders
        )
        self.diagnostics_logger = DiagnosticsLogger(
            logger=self.logger,
            num_classes=self.num_classes,
            criterion=self.criterion,
            diagnostics_config=diagnostics_config,
            context=diagnostics_context,
        )
        self.snapshots = self.diagnostics_logger.snapshots

    @staticmethod
    def _capture_rng_state():
        rng_state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state['torch_cuda'] = torch.cuda.get_rng_state_all()
        return rng_state

    @staticmethod
    def _restore_rng_state(rng_state):
        if not rng_state:
            return

        python_state = rng_state.get('python')
        numpy_state = rng_state.get('numpy')
        torch_state = rng_state.get('torch')
        torch_cuda_state = rng_state.get('torch_cuda')

        if python_state is not None:
            random.setstate(python_state)
        if numpy_state is not None:
            np.random.set_state(numpy_state)
        if torch_state is not None:
            torch.set_rng_state(torch_state)
        if torch_cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(torch_cuda_state)

    def resume(self, resume_path):
        if os.path.isfile(resume_path):
            self.logger.info(("=> loading checkpoint '{}'".format(resume_path)))
            checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
            self.resume_checkpoint = checkpoint
            self.start_epoch = int(checkpoint['epoch'])
            self.best_acc = checkpoint['best_acc']
            self.best_epoch = checkpoint['best_epoch']
            self.total_step = int(checkpoint.get('total_step', 0))
            self.total_time = float(checkpoint.get('total_time', 0.0))
            self.time_this_epoch = float(checkpoint.get('time_this_epoch', 0.0))
            # self.model.load_state_dict(checkpoint['state_dict'])
            self.model.module.load_state_dict(checkpoint['state_dict']) if hasattr(self.model, 'module') else self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self._restore_rng_state(checkpoint.get('rng_state'))
            self.logger.info(("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch'])))
        else:
            self.logger.info(("=> no checkpoint found at '{}'".format(resume_path)))

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        filename = os.path.join(self.config['save_dir'],filename)
        best_filename = os.path.join(self.config['save_dir'],'model_best.pth.tar')
        torch.save(state, filename)
        # self.logger.info(f'Save checkpoint to {filename}')
        if is_best:
            shutil.copyfile(filename, best_filename)
            # self.logger.info(f'Save best checkpoint to {best_filename}')

    def run(self):
        self.before_run()
        self.run_begin_time = time.time() - self.total_time
        self.logger.info(f'Begin training for {self.method_name}...')

        if self.total_step == 0:
            self.compute_diagnostics(trigger='run_start', epoch=self.start_epoch, batch_idx=-1, force=True)
        else:
            self.logger.info(
                f"=====> Resuming training at epoch {self.start_epoch + 1}, total_step {self.total_step}, "
                f"best_epoch {self.best_epoch}, best_acc {self.best_acc:.4f}"
            )
        for epoch in range(self.start_epoch+1, self.epochs+1):
            self.before_epoch(epoch)
            self.train(epoch)
            self.after_epoch(epoch)
            if self.num_steps is not None and self.total_step >= self.num_steps:
                self.logger.info(f'Finish training for {self.method_name} because num_steps {self.num_steps} is reached')
                break

        self.after_run()

    def before_run(self):
        self.total_time = float(getattr(self, 'total_time', 0.0))
        self.time_this_epoch = float(getattr(self, 'time_this_epoch', 0.0))
        self.total_step = int(getattr(self, 'total_step', 0))

    def before_epoch(self, epoch):
        # select samples for this epoch
        return

    def after_epoch(self, epoch):
        self.save_model(epoch)
        self.diagnostics_logger.log_epoch_end_selection_stats(
            epoch=epoch,
            total_step=self.total_step,
        )
        return

    def after_run(self):
        self.diagnostics_logger.finalize()

    def before_batch(self, i, inputs, targets, indexes, epoch) -> MinibatchInfo:
        # online batch selection
        return MinibatchInfo(inputs, targets, indexes)

    def after_batch(self, i, inputs, targets, indexes, outputs, epoch):
        self.total_step += 1
        # self.ema_net.update()

        self.compute_diagnostics(
            trigger='batch_update',
            epoch=epoch,
            batch_idx=i,
            indexes=indexes,
        )

    def _build_checkpoint_state(self, epoch):
        return {
            'epoch': epoch,
            'total_step': self.total_step,
            'total_time': self.total_time,
            'time_this_epoch': self.time_this_epoch,
            'state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'best_epoch': self.best_epoch,
            'wandb_run_id': self.config.get('wandb_run_id'),
            'rng_state': self._capture_rng_state(),
        }

    def compute_diagnostics(self, trigger, epoch, batch_idx, indexes=None, force=False):
        model = self.model.module if hasattr(self.model, 'module') else self.model
        device = next(model.parameters()).device

        diagnostics_state = self.diagnostics_logger.log_diagnostics(
            model=model,
            trigger=trigger,
            total_step=self.total_step,
            epoch=epoch,
            batch_idx=batch_idx,
            device=device,
            selected_indexes=indexes,
            lr=self.optimizer.param_groups[0]['lr'],
            total_time=self.total_time,
            time_this_epoch=self.time_this_epoch,
            checkpoint_state=self._build_checkpoint_state(epoch),
            force=force,
        )
        self.best_acc = diagnostics_state['best_acc']
        self.best_epoch = diagnostics_state['best_epoch']
        self.is_best = diagnostics_state['is_best']

    def log_step_param_grad_stats(self, epoch):
        model = self.model.module if hasattr(self.model, 'module') else self.model
        self.diagnostics_logger.log_step_param_grad_stats(
            model=model,
            total_step=self.total_step + 1,
            epoch=epoch,
        )


    def train(self, epoch):
        # train for one epoch and record time taken
        total_batch = len(self.train_loader)
        epoch_begin_time = time.time()
        
        # train
        for i, datas in enumerate(self.train_loader):
            self.model.train()
            metabatch_inputs = datas["input"].cuda()
            metabatch_targets = datas["target"].cuda()
            metabatch_indexes = datas["index"]
            minibatch = self.before_batch(
                i, metabatch_inputs, metabatch_targets, metabatch_indexes, epoch
            )
            selected_outputs, features = (
                self.model(
                    x=minibatch.inputs,
                    need_features=self.need_features,
                    targets=minibatch.targets,
                )
                if self.need_features
                else (
                    self.model(
                        x=minibatch.inputs,
                        need_features=False,
                        targets=minibatch.targets,
                    ),
                    None,
                )
            )
            
            # Reweight loss using minibatch weights, if present
            criterion_reduction = 'mean' if minibatch.weights is None else 'weighted'
            loss = self.criterion(
                selected_outputs, 
                minibatch.targets, 
                reduction=criterion_reduction,
                weights=minibatch.weights  # If None, not applied
            )

            self.while_update(
                selected_outputs,
                loss,
                minibatch.targets,
                epoch,
                features,
                minibatch.indices,
                batch_idx=i,
                batch_size=self.batch_size,
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.log_step_param_grad_stats(epoch)
            self.optimizer.step()
            self.after_batch(
                i,
                minibatch.inputs,
                minibatch.targets,
                minibatch.indices,
                selected_outputs.detach(),
                epoch,
            )

        now = time.time()
        self.time_this_epoch = now - epoch_begin_time
        self.total_time = now - self.run_begin_time
        self.scheduler.step()

    def while_update(self, outputs, loss, targets, epoch, features, indexes, batch_idx, batch_size):
        pass

    def save_model(self, epoch):
        # save model
        self.logger.info('=====> Saving current model and best model')
        checkpoint = self._build_checkpoint_state(epoch)
        self.save_checkpoint(checkpoint, self.is_best)
    


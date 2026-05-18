import os

import numpy as np
import torch
import torch.nn.functional as F

from methods.method_utils.diagnostics_context import DiagnosticsRunContext


class SnapshotManager:
    def __init__(
        self,
        logger,
        context: DiagnosticsRunContext,
        log_interval,
        save_init,
        save_freq,
        should_track_points,
        should_save_snapshots,
        should_save_best_checkpoint,
        initial_best_acc=0.0,
        initial_best_epoch=0,
        checkpoint_saver=None,
    ):
        self.logger = logger
        self.context = context
        self.fixed_train_loader = context.fixed_train_loader
        self.test_loader = context.test_loader
        self.total_batches = int(context.total_batches)
        self.num_train_samples = int(context.num_train_samples)
        self.num_epochs = context.num_epochs
        self.num_steps = context.num_steps
        self.log_interval = log_interval
        self.save_init = int(save_init)
        self.save_freq = int(save_freq)
        self.should_track_points = should_track_points
        self.should_save_snapshots = should_save_snapshots
        self.should_save_best_checkpoint = should_save_best_checkpoint
        self.best_acc = float(initial_best_acc if initial_best_acc is not None else context.initial_best_acc)
        self.best_epoch = int(initial_best_epoch if initial_best_epoch is not None else context.initial_best_epoch)
        self.checkpoint_saver = checkpoint_saver if checkpoint_saver is not None else context.checkpoint_saver
        self.true_labels = self._to_long_tensor(context.true_labels)
        if self.true_labels is not None and self.true_labels.numel() != self.num_train_samples:
            raise ValueError(
                f'true_labels has length {self.true_labels.numel()}, expected {self.num_train_samples}.'
            )
        self.noisy_indices = self._to_numpy_indices(context.noisy_indices)
        self.current_epoch = None
        self.current_epoch_selected_mask = None
        if self.noisy_indices is not None and self.noisy_indices.size > 0:
            self.current_epoch_selected_mask = np.zeros(self.num_train_samples, dtype=np.uint8)

        if self.log_interval not in {'logarithmic', 'per_epoch'}:
            self.logger.info(
                f"Warning: unknown diagnostics log_interval '{self.log_interval}'. Falling back to 'per_epoch'."
            )
            self.log_interval = 'per_epoch'

        self.last_batch_idx = self.total_batches - 1
        self._logged_steps = set()
        self._logarithmic_steps = self._build_logarithmic_steps()

        self.snapshots = []
        self.snapshot_steps = []
        snapshots_dir = os.path.join(self.context.project_root, 'snapshots', self.context.dataset_name)
        os.makedirs(snapshots_dir, exist_ok=True)
        self.snapshots_path = os.path.join(snapshots_dir, f'{self.context.artifact_stem}.p')

        self.selected_points_by_epoch = None
        self.selected_points_path = None
        if self.should_track_points:
            self._initialize_selected_points_tracking()

    @staticmethod
    def _to_long_tensor(values):
        if values is None:
            return None
        if isinstance(values, torch.Tensor):
            return values.detach().cpu().long().reshape(-1)
        return torch.as_tensor(values, dtype=torch.long).reshape(-1)

    @staticmethod
    def _to_numpy_indices(values):
        if values is None:
            return None
        if isinstance(values, torch.Tensor):
            return values.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
        return np.asarray(values, dtype=np.int64).reshape(-1)

    def _build_logarithmic_steps(self):
        total_epochs = self.num_epochs or int(np.ceil(self.num_steps / self.total_batches))
        intra_epoch_stride = max(self.total_batches // self.save_freq, 1)

        t = 0
        steps = [0]
        for epoch in range(total_epochs):
            for batch_idx in range(self.total_batches):
                t += 1
                if epoch < self.save_init and batch_idx % intra_epoch_stride == 0:
                    steps.append(t)

            if self.save_init <= epoch <= 25:
                steps.append(t)
            elif 25 < epoch <= 65 and epoch % 4 == 0:
                steps.append(t)
            elif (epoch > 65 and epoch % 15 == 0) or (epoch == total_epochs - 1):
                steps.append(t)

        if self.num_steps is not None:
            steps = [step for step in steps if step <= self.num_steps]

        return set(steps)

    def _initialize_selected_points_tracking(self):
        num_tracking_epochs = self.num_epochs or int(np.ceil(self.num_steps / self.total_batches))
        self.selected_points_by_epoch = np.zeros(
            (num_tracking_epochs, self.num_train_samples),
            dtype=np.uint8,
        )
        selected_points_dir = os.path.join(self.context.project_root, 'selected_points', self.context.dataset_name)
        os.makedirs(selected_points_dir, exist_ok=True)
        self.selected_points_path = os.path.join(
            selected_points_dir,
            f'{self.context.artifact_stem}.npy',
        )

    def should_log(self, total_step, batch_idx, force=False):
        if force:
            return True
        if total_step in self._logged_steps:
            return False
        if self.log_interval == 'per_epoch':
            return batch_idx == self.last_batch_idx
        return total_step in self._logarithmic_steps

    def mark_logged(self, total_step):
        self._logged_steps.add(total_step)

    def record_selected_points(self, epoch, indexes):
        if indexes is None:
            return

        epoch_idx = epoch - 1
        if isinstance(indexes, torch.Tensor):
            selected_indexes = indexes.detach().cpu().numpy()
        else:
            selected_indexes = np.asarray(indexes)

        selected_indexes = np.asarray(selected_indexes, dtype=np.int64).reshape(-1)
        if selected_indexes.size == 0:
            return

        valid_mask = (selected_indexes >= 0) & (selected_indexes < self.num_train_samples)
        selected_indexes = selected_indexes[valid_mask]
        if selected_indexes.size == 0:
            return

        if self.should_track_points:
            self.selected_points_by_epoch[epoch_idx, selected_indexes] = 1

        if self.current_epoch_selected_mask is not None:
            if self.current_epoch != int(epoch):
                self.current_epoch_selected_mask.fill(0)
                self.current_epoch = int(epoch)
            self.current_epoch_selected_mask[selected_indexes] = 1

    def get_percent_noisy_selected(self):
        if self.current_epoch_selected_mask is None:
            return None
        return float(self.current_epoch_selected_mask[self.noisy_indices].sum() / self.num_train_samples)

    def get_noisy_selection_stats(self):
        if self.current_epoch_selected_mask is None:
            return None

        num_noisy_selected = int(self.current_epoch_selected_mask[self.noisy_indices].sum())
        fraction_of_train = float(num_noisy_selected / self.num_train_samples)
        total_noisy = int(self.noisy_indices.size)
        fraction_of_noisy_pool = float(num_noisy_selected / total_noisy) if total_noisy > 0 else 0.0
        return {
            'num_noisy_selected': num_noisy_selected,
            'fraction_of_train': fraction_of_train,
            'fraction_of_noisy_pool': fraction_of_noisy_pool,
            'total_noisy': total_noisy,
        }

    def uses_true_labels_for_train_accuracy(self):
        return self.true_labels is not None

    def _calculate_snapshot_stats(self, model, loader, device, true_labels=None):
        log_probs = []
        losses = []
        errors = []

        for datas in loader:
            inputs = datas['input'].to(device)
            targets = datas['target'].to(device)
            batch_log_probs = F.log_softmax(model(inputs), dim=1)
            batch_losses = -torch.gather(batch_log_probs, 1, targets.view(-1, 1))
            predicted = torch.argmax(batch_log_probs, dim=1).long()
            eval_targets = targets
            if true_labels is not None:
                indexes = datas['index']
                if not isinstance(indexes, torch.Tensor):
                    indexes = torch.as_tensor(indexes, dtype=torch.long)
                eval_targets = true_labels[indexes.detach().cpu()].to(device)
            batch_errors = (eval_targets != predicted).float()
            log_probs.append(batch_log_probs.detach().cpu())
            losses.append(batch_losses.detach().cpu())
            errors.append(batch_errors.detach().cpu())

        return (
            torch.cat(log_probs, dim=0),
            torch.cat(losses, dim=0),
            torch.cat(errors, dim=0),
        )

    def build_snapshot(self, model, device, total_step, epoch):
        # yh: predicted_probs
        # f: list of losses
        # e: list of errors (1 - acc)
        with torch.no_grad():
            yh, f, e = self._calculate_snapshot_stats(
                model,
                self.fixed_train_loader,
                device,
                true_labels=self.true_labels,
            )
            yvh, fv, ev = self._calculate_snapshot_stats(model, self.test_loader, device)

        snapshot = {
            'yh': yh,
            'f': f,
            'e': e,
            'yvh': yvh,
            'fv': fv,
            'ev': ev,
            'total_step': int(total_step),
            'epoch': int(epoch),
        }
        metrics = {
            'train_loss': float(f.mean().item()),
            'train_acc': float(1.0 - e.mean().item()),
            'val_loss': float(fv.mean().item()),
            'val_acc': float(1.0 - ev.mean().item()),
            'train_normed_logits_l2_mean': float(torch.norm(yh, p=2, dim=1).mean().item()),
            'val_normed_logits_l2_mean': float(torch.norm(yvh, p=2, dim=1).mean().item()),
        }
        return snapshot, metrics

    def store_snapshot(self, snapshot, total_step):
        if not self.should_save_snapshots:
            return
        self.snapshots.append(snapshot)
        self.snapshot_steps.append(int(total_step))

    def log_summary(self, epoch, total_step, lr, total_time, time_this_epoch, metrics):
        total_epoch_display = self.num_epochs or '?'
        train_loss_label = 'Train Loss (train loader labels)'
        train_acc_label = 'Train acc (true labels)' if self.uses_true_labels_for_train_accuracy() else 'Train acc (train loader labels)'
        self.logger.info(
            f'=====> Epoch: {epoch}/{total_epoch_display}, Total_step: {total_step}, '
            f'lr: {lr:.6f}, Total Time: {total_time:.4f} s, '
            f'Time this epoch: {time_this_epoch:.4f} s'
        )
        self.logger.info(
            f'=====> Epoch: {epoch}/{total_epoch_display}, '
            f'{train_loss_label}: {metrics["train_loss"]:.4f}, {train_acc_label}: {metrics["train_acc"]:.4f}, '
            f'Best val acc: {self.best_acc:.4f}, Current val acc: {metrics["val_acc"]:.4f}, '
            f'Val Loss: {metrics["val_loss"]:.4f}'
        )

    def update_best_checkpoint(self, epoch, metrics, checkpoint_state):
        is_best = False
        if metrics['val_acc'] > self.best_acc:
            self.best_acc = metrics['val_acc']
            self.best_epoch = int(epoch)
            is_best = True

        if not is_best:
            return False

        if self.should_save_best_checkpoint and self.checkpoint_saver is not None:
            checkpoint_state = dict(checkpoint_state)
            checkpoint_state['best_acc'] = self.best_acc
            checkpoint_state['best_epoch'] = self.best_epoch
            self.checkpoint_saver(checkpoint_state, True)
        return True

    def _save_selected_points(self):
        np.save(self.selected_points_path, self.selected_points_by_epoch)
        self.logger.info(f'Saved selected points array to {self.selected_points_path}')

    def _save_snapshots(self):
        payload = {
            'data': self.snapshots,
            'steps': self.snapshot_steps,
        }
        torch.save(payload, self.snapshots_path)
        self.logger.info(f'Saved helper snapshots to {self.snapshots_path}')

    def finalize(self):
        if self.should_save_snapshots:
            self._save_snapshots()
        if self.should_track_points:
            self._save_selected_points()
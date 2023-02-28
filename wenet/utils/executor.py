# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from contextlib import nullcontext

# if your python version < 3.7 use the below one
# from contextlib import suppress as nullcontext
import torch, os
from torch.nn.utils import clip_grad_norm_
from wenet.utils.checkpoint import save_checkpoint
from wenet.utils.ctc_kd_loss import CTCKDLoss
from wenet.utils.mwer import CTCMWERLoss

class Executor:

    def __init__(self):
        self.step = 0

    def train(self, model, optimizer, scheduler, data_loader, device, writer,
              args, scaler, teacher_model=None, cv_data_loader=None):
        ''' Train one epoch
        '''
        model.train()
        clip = args.get('grad_clip', 50.0)
        log_interval = args.get('log_interval', 10)
        rank = args.get('rank', 0)
        epoch = args.get('epoch', 0)
        accum_grad = args.get('accum_grad', 1)
        is_distributed = args.get('is_distributed', True)
        use_amp = args.get('use_amp', False)
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(accum_grad))
        save_ckpt_steps = args.get("save_ckpt_steps", 5000)
        model_dir = args.get("model_dir", None)
        if use_amp:
            assert scaler is not None
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_context = model.join
        else:
            model_context = nullcontext
        num_seen_utts = 0
        with model_context():
            for batch_idx, batch in enumerate(data_loader):
                key, feats, target, feats_lengths, target_lengths = batch
                feats = feats.to(device)
                target = target.to(device)
                feats_lengths = feats_lengths.to(device)
                target_lengths = target_lengths.to(device)
                num_utts = target_lengths.size(0)
                if num_utts == 0:
                    continue
                context = None
                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if is_distributed and batch_idx % accum_grad != 0:
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext
                with context():
                    # autocast context
                    # The more details about amp can be found in
                    # https://pytorch.org/docs/stable/notes/amp_examples.html
                    with torch.cuda.amp.autocast(scaler is not None):
                        loss_dict = model(feats, feats_lengths, target, target_lengths)
                        # get distillation loss
                        if not teacher_model is None:
                            encoder_out, encoder_out_lens = loss_dict['encoder_out'], loss_dict['encoder_out_lens']
                            t_encoder_out, t_encoder_out_mask = teacher_model.encoder(feats, feats_lengths)

                            # # distill the encoder_out directly
                            distill_loss = torch.nn.functional.kl_div(encoder_out.softmax(dim=-1).log(),
                                                                        t_encoder_out.softmax(dim=-1),
                                                                        reduction='batchmean')

                            t_encoder_out_lens = t_encoder_out_mask.squeeze(1).sum(1)
                            s_logits = model.ctc.log_softmax(encoder_out)
                            t_logits = teacher_model.ctc.log_softmax(t_encoder_out)
                            # # distill CTC logits with the teacher's decoding result.
                            # distill_loss = CTCKDLoss(nbest=2).forward(s_logits=s_logits, s_logits_length=encoder_out_lens,
                            #             t_logits=t_logits, t_logits_length=t_encoder_out_lens)

                            loss_dict['loss_distill'] = distill_loss
                            teacher_distill_weight = args.get('teacher_distill_weight', 0)
                            total_loss = (1 - teacher_distill_weight) * loss_dict['loss'] + teacher_distill_weight * distill_loss
                            loss_dict['loss'] = total_loss

                        # get mwer loss
                        mwer_weight = args.get('mwer_weight', 0)
                        if mwer_weight > 0:
                            encoder_out, encoder_out_lens = loss_dict['encoder_out'], loss_dict['encoder_out_lens']
                            logits = model.ctc.log_softmax(encoder_out)
                            mwer_loss = CTCMWERLoss(beam_width=4).forward(logits=logits, logit_length=encoder_out_lens,
                                            labels=target, label_length=target_lengths)
                            loss_dict['loss_mwer'] = mwer_loss
                            total_loss = (1 - mwer_weight) * loss_dict['loss'] + mwer_weight * mwer_loss
                            loss_dict['loss'] = total_loss

                        # gradient accumulate
                        loss = loss_dict['loss'] / accum_grad
                    if use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                num_seen_utts += num_utts
                if batch_idx % accum_grad == 0:
                    if rank == 0 and writer is not None:
                        writer.add_scalar('train_loss', loss, self.step)
                    # Use mixed precision training
                    if use_amp:
                        scaler.unscale_(optimizer)
                        grad_norm = clip_grad_norm_(model.parameters(), clip)
                        # Must invoke scaler.update() if unscale_() is used in
                        # the iteration to avoid the following error:
                        #   RuntimeError: unscale_() has already been called
                        #   on this optimizer since the last update().
                        # We don't check grad here since that if the gradient
                        # has inf/nan values, scaler.step will skip
                        # optimizer.step().
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        grad_norm = clip_grad_norm_(model.parameters(), clip)
                        if torch.isfinite(grad_norm):
                            optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    self.step += 1

                    if self.step % save_ckpt_steps == 0:
                        lr = optimizer.param_groups[0]['lr']
                        if not cv_data_loader is None:
                            total_loss, num_seen_utts = self.cv(model, cv_data_loader, device, args)
                            cv_loss = total_loss / num_seen_utts
                            logging.info('Epoch {} Step {} CV info cv_loss {}'.format(epoch, self.step, cv_loss))
                            if model_dir is not None and rank == 0:
                                logging.debug(f"Saved ckpt {epoch} epoch, {self.step} steps.")
                                save_model_path = os.path.join(model_dir, 'steps_{}.pt'.format(self.step))
                                save_checkpoint(model, save_model_path,
                                                {
                                                    'epoch': epoch,
                                                    'lr': lr,
                                                    'cv_loss': cv_loss,
                                                    'step': self.step
                                                })
                                writer.add_scalar('cv_loss', cv_loss, self.step)
                                writer.add_scalar('lr', lr, self.step)
                            model.train()
                        else:
                            if model_dir is not None and rank == 0:
                                logging.debug(f"Saved ckpt {epoch} epoch, {self.step} steps.")
                                save_model_path = os.path.join(model_dir, 'steps_{}.pt'.format(self.step))
                                save_checkpoint(model, save_model_path,
                                                {
                                                    'epoch': epoch,
                                                    'lr': lr,
                                                    'step': self.step
                                                })
                                writer.add_scalar('lr', lr, self.step)
                if batch_idx % log_interval == 0:
                    lr = optimizer.param_groups[0]['lr']
                    log_str = 'TRAIN Batch {}/{} loss {:.6f} '.format(
                        epoch, batch_idx,
                        loss.item() * accum_grad)
                    for name, value in loss_dict.items():
                        if name.startswith('loss_') and value is not None:
                            log_str += '{} {:.6f} '.format(name, value.item())
                    log_str += 'lr {:.8f} rank {}'.format(lr, rank)
                    logging.debug(log_str)


    def cv(self, model, data_loader, device, args):
        ''' Cross validation on
        '''
        model.eval()
        rank = args.get('rank', 0)
        epoch = args.get('epoch', 0)
        log_interval = args.get('log_interval', 10)
        # in order to avoid division by 0
        num_seen_utts = 1
        total_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                key, feats, target, feats_lengths, target_lengths = batch
                feats = feats.to(device)
                target = target.to(device)
                feats_lengths = feats_lengths.to(device)
                target_lengths = target_lengths.to(device)
                num_utts = target_lengths.size(0)
                if num_utts == 0:
                    continue
                loss_dict = model(feats, feats_lengths, target, target_lengths)
                loss = loss_dict['loss']
                if torch.isfinite(loss):
                    num_seen_utts += num_utts
                    total_loss += loss.item() * num_utts
                if batch_idx % log_interval == 0:
                    log_str = 'CV Batch {}/{} loss {:.6f} '.format(
                        epoch, batch_idx, loss.item())
                    for name, value in loss_dict.items():
                        if name.startswith('loss_') and value is not None:
                            log_str += '{} {:.6f} '.format(name, value.item())
                    log_str += 'history loss {:.6f}'.format(total_loss /
                                                            num_seen_utts)
                    log_str += ' rank {}'.format(rank)
                    logging.debug(log_str)
        return total_loss, num_seen_utts

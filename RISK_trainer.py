from torch.optim import lr_scheduler

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, save_to_csv, visual_weights
from utils.metrics import metric, MAE, MSE
from utils.metrics import DMSE

from models.Risk_core import RiskCore

import torch
import torch.nn as nn
from torch import optim
import math
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        if self.args.model == 'RATE':
            model = RiskCore(self.args).float()
        else:
            model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            if len(self.args.device_ids) > 1:
                print(f'Using {len(self.args.device_ids)} GPUs: {self.args.device_ids}')
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.data == 'PEMS':
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        all_preds = []
        all_trues = []
        all_gates = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None

                if self.args.model == 'RATE':
                    outputs_dict = self.model(batch_x, batch_x_mark, y_true=batch_y)
                    
                    loss = outputs_dict['total_loss'].mean()
                    outputs = outputs_dict['y_final']
                    
                    gates = outputs_dict['gate'].detach().cpu().numpy()
                    all_gates.append(gates)

                total_loss.append(loss.item())
                pred_np = outputs.detach().cpu().numpy()
                true_np = batch_y.detach().cpu().numpy()
                all_preds.append(pred_np)
                all_trues.append(true_np)

        total_loss = np.average(total_loss)
        
        if len(all_preds) > 0:
            all_preds = np.concatenate(all_preds, axis=0)
            all_trues = np.concatenate(all_trues, axis=0)
            val_mae = MAE(all_preds, all_trues)
            val_mse = MSE(all_preds, all_trues)
            
            if self.args.model == 'RATE' and all_gates:
                all_gates = np.concatenate(all_gates, axis=0)
                avg_gate = np.mean(all_gates)
                print(f"\tValidation MAE: {val_mae:.7f}, MSE: {val_mse:.7f} | gate: {avg_gate:.3f}")
            else:
                print(f"\tValidation MAE: {val_mae:.7f}, MSE: {val_mse:.7f}")
        
        self.model.train()
        return total_loss, val_mae, val_mse

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # Learning rate scheduler (optional OneCycle)
        scheduler = None
        if int(getattr(self.args, 'use_onecycle', 1)) == 1:
            steps_per_epoch = math.ceil(train_steps / max(1, int(getattr(self.args, 'accumulate', 1))))
            scheduler = lr_scheduler.OneCycleLR(
                optimizer=model_optim,
                steps_per_epoch=steps_per_epoch,
                pct_start=self.args.pct_start,
                epochs=self.args.train_epochs,
                max_lr=self.args.learning_rate
            )

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            epoch_gate_values = []  

            self.model.train()
            epoch_time = time.time()
            accumulate_steps = max(1, int(getattr(self.args, 'accumulate', 1)))
            clip_max_norm = float(getattr(self.args, 'clip_grad_norm', 0.0))
            model_optim.zero_grad()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None

                if self.args.model == 'RATE':

                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs_dict = self.model(batch_x, batch_x_mark, y_true=batch_y)
                            loss = outputs_dict['total_loss'].mean()
                    else:
                        outputs_dict = self.model(batch_x, batch_x_mark, y_true=batch_y)
                        loss = outputs_dict['total_loss'].mean()
                    
                    avg_gate = outputs_dict['gate'].mean().item()
                    epoch_gate_values.append(avg_gate)
                    
                    train_loss.append(loss.item())
                    
                    if (i + 1) % 100 == 0:
                        with torch.no_grad():
                            pred_np = outputs_dict['y_final'].detach().cpu().numpy()
                            true_np = batch_y.detach().cpu().numpy()
                            batch_mae = MAE(pred_np, true_np)
                            batch_mse = MSE(pred_np, true_np)
                            avg_gates = np.mean(epoch_gate_values[-100:])  
                        
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f} | MAE: {3:.7f}, MSE: {4:.7f} | gate: {5:.3f}".format(
                            i + 1, epoch + 1, loss.item(), batch_mae, batch_mse, avg_gates))
                        

                        def get_loss(name, default=0.0):
                            v = outputs_dict.get(name, None)
                            if v is None:
                                return default
                            try:
                                return float(v.mean().item())
                            except Exception:
                                return default

                        pred_loss = get_loss('pred_loss')
                        residual_loss = get_loss('residual_loss')
                        hf_penalty = get_loss('hf_penalty')
                        lf_penalty = get_loss('lf_penalty')
                        comp_loss = get_loss('comp_loss')
                        gate_loss = get_loss('gate_loss')
                        tv_loss = get_loss('tv_loss')

                        print("\t  pred:{:.7f} | res:{:.7f} | hf:{:.7f} | lf:{:.7f} | comp:{:.7f} | gate:{:.7f} | tv:{:.7f}".format(
                            pred_loss, residual_loss, hf_penalty, lf_penalty, comp_loss, gate_loss, tv_loss))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()
                
                else:
                    if self.args.down_sampling_layers == 0:
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    else:
                        dec_inp = None

                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            loss = criterion(outputs, batch_y)
                            train_loss.append(loss.item())
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0

                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())

                    if (i + 1) % 100 == 0:
                        with torch.no_grad():
                            if self.args.use_amp:
                                outputs_eval = outputs[:, -self.args.pred_len:, f_dim:]
                                batch_y_eval = batch_y[:, -self.args.pred_len:, f_dim:]
                            else:
                                outputs_eval = outputs[:, -self.args.pred_len:, f_dim:]
                                batch_y_eval = batch_y[:, -self.args.pred_len:, f_dim:]
                            
                            pred_np = outputs_eval.detach().cpu().numpy()
                            true_np = batch_y_eval.detach().cpu().numpy()
                            batch_mae = MAE(pred_np, true_np)
                            batch_mse = MSE(pred_np, true_np)
                        
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f} | MAE: {3:.7f}, MSE: {4:.7f}".format(
                            i + 1, epoch + 1, loss.item(), batch_mae, batch_mse))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss / accumulate_steps).backward()
                else:
                    (loss / accumulate_steps).backward()

                do_step = ((i + 1) % accumulate_steps == 0) or (i + 1 == train_steps)
                if do_step:
                    if clip_max_norm > 0:
                        if self.args.use_amp:
                            scaler.unscale_(model_optim)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_max_norm)

                    if self.args.use_amp:
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        model_optim.step()
                    model_optim.zero_grad()

                    if scheduler is not None and self.args.lradj == 'TST':
                        scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            
            if self.args.model == 'RATE' and epoch_gate_values:
                avg_epoch_gate_values = np.mean(epoch_gate_values)
                print("Epoch {0} average gate: {1:.3f}".format(epoch + 1, avg_epoch_gate_values))
            
            vali_loss, vali_mae, vali_mse = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_mae, test_mse = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(test_mae + test_mse, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if scheduler is not None and self.args.lradj == 'TST':
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
            elif self.args.lradj in ['type1', 'type2', 'type3', 'PEMS']:
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=True)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        checkpoints_path = './checkpoints/' + setting + '/'
        preds = []
        trues = []
        all_gates = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None

                if self.args.model == 'RATE':
                    y_final, y_stable, y_residual, risk_pred, gate, decomposed_level, decomposed_season, decomposed_trend, decomposed_residual = self.model(batch_x, batch_x_mark)
                    variant = getattr(self.args, 'risk_variant', 'fused')
                    if variant == 'stable-only':
                        outputs = y_stable
                    elif variant == 'residual-only':
                        outputs = y_residual
                    else:
                        outputs = y_final
                    
                    all_gates.append(gate.detach().cpu().numpy())

                else:
                    if self.args.down_sampling_layers == 0:
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    else:
                        dec_inp = None

                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        if self.args.data == 'PEMS':
            B, T, C = preds.shape
            preds = test_data.inverse_transform(preds.reshape(-1, C)).reshape(B, T, C)
            trues = test_data.inverse_transform(trues.reshape(-1, C)).reshape(B, T, C)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        dmse_strength = getattr(self.args, 'dmse_strength', 4.0)
        dmse = DMSE(preds, trues, ref=trues, strength=dmse_strength)
        print('mse:{}, mae:{}, dmse:{} (strength={})'.format(mse, mae, dmse, dmse_strength))

        avg_gate = None
        if self.args.model == 'RATE' and all_gates:
            all_gates = np.concatenate(all_gates, axis=0)
            avg_gate = np.mean(all_gates)
            print(f"Test gate: {avg_gate:.3f}")

        result_file = os.path.join(folder_path, 'result.txt')
        with open(result_file, 'w') as f:
            f.write(f'mse:{mse}, mae:{mae}, dmse:{dmse}\n')
            if avg_gate is not None:
                f.write(f'gate:{avg_gate:.3f}\n')
        return

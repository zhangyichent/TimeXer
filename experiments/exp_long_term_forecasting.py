from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.timefeatures import convert_inttime_to_strtime,get_now
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import traceback

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion


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

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (seq_exogenous_x, seq_endogenous_x, seq_endogenous_y, seq_x_mark, seq_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                seq_exogenous_x = seq_exogenous_x.float().to(self.device)
                seq_endogenous_x = seq_endogenous_x.float().to(self.device)
                seq_endogenous_y = seq_endogenous_y.float().to(self.device)
                seq_x_mark = seq_x_mark.float().to(self.device)
                seq_y_mark = seq_y_mark.float().to(self.device)

                # decoder input
                # dec_inp = torch.zeros_like(seq_endogenous_y[:, -self.args.pred_len:, :]).float()#先把后面预测的位置空下来
                # dec_seq_endogenous_y = torch.cat([seq_endogenous_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)# 这个和iTransformer无关，所以其实是没用代码

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(seq_exogenous_x, seq_endogenous_x, seq_x_mark)[0]
                        else:
                            outputs = self.model(seq_exogenous_x, seq_endogenous_x, seq_x_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        seq_endogenous_y = seq_endogenous_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, seq_endogenous_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(seq_exogenous_x, seq_endogenous_x, seq_x_mark)[0]
                    else:
                        outputs = self.model(seq_exogenous_x, seq_endogenous_x, seq_x_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    seq_endogenous_y = seq_endogenous_y[:, -self.args.pred_len:, f_dim:].to(self.device)#对于TimeXer目前只有一个变量，所以f_dim是啥都无所谓
                    loss = criterion(outputs, seq_endogenous_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model
        
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        preds = []#
        trues = []#
        print('HERE in vali!!!!!!!!!!!!!!!')
        with torch.no_grad():
            for i, (seq_exogenous_x, seq_endogenous_x, seq_endogenous_y, seq_x_mark, seq_y_mark) in enumerate(vali_loader):
                seq_exogenous_x = seq_exogenous_x.float().to(self.device)
                seq_endogenous_x = seq_endogenous_x.float().to(self.device)
                seq_endogenous_y = seq_endogenous_y.float().to(self.device)
                seq_x_mark = seq_x_mark.float().to(self.device)
                seq_y_mark = seq_y_mark.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = \
                            self.model(seq_exogenous_x, seq_endogenous_x, seq_x_mark)[0]
                        else:
                            outputs = self.model(seq_exogenous_x, seq_endogenous_x, seq_x_mark)
                else:
                    if self.args.output_attention:
                        outputs = \
                            self.model(seq_exogenous_x, seq_endogenous_x, seq_x_mark)[0]
                    else:
                        outputs = self.model(seq_exogenous_x, seq_endogenous_x, seq_x_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                seq_endogenous_y = seq_endogenous_y[:, -self.args.pred_len:, :].to(self.device)

                pred = outputs.detach().cpu()#.numpy()
                true = seq_endogenous_y.detach().cpu()#.numpy()
                pred = pred[:, :, f_dim:]
                true = true[:, :, f_dim:]

                preds.append(pred)#
                trues.append(true)#

                loss = criterion(pred, true)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        print('total_loss:',total_loss)
        self.model.train()
        return total_loss
        
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (seq_exogenous_x, seq_endogenous_x, seq_endogenous_y, seq_x_mark, seq_y_mark) in enumerate(test_loader):
                seq_exogenous_x = seq_exogenous_x.float().to(self.device)
                seq_endogenous_x = seq_endogenous_x.float().to(self.device)
                seq_endogenous_y = seq_endogenous_y.float().to(self.device)
                seq_x_mark = seq_x_mark.float().to(self.device)
                seq_y_mark = seq_y_mark.float().to(self.device)

                # # decoder input
                # dec_inp = torch.zeros_like(seq_endogenous_y[:, -self.args.pred_len:, :]).float()
                # dec_seq_endogenous_y = torch.cat([seq_endogenous_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = \
                                self.model(seq_exogenous_x, seq_endogenous_x, seq_x_mark)[0]
                        else:
                            outputs = self.model(seq_exogenous_x, seq_endogenous_x, seq_x_mark)
                else:
                    if self.args.output_attention:
                        outputs = \
                            self.model(seq_exogenous_x, seq_endogenous_x, seq_x_mark)[0]
                    else:
                        outputs = self.model(seq_exogenous_x, seq_endogenous_x, seq_x_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                seq_endogenous_y = seq_endogenous_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                seq_endogenous_y = seq_endogenous_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                     shape = outputs.shape
                     outputs = test_data.inverse_endogenous_transform(outputs.squeeze(0)).reshape(shape)
                     seq_endogenous_y = test_data.inverse_endogenous_transform(seq_endogenous_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = seq_endogenous_y

                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    input = seq_endogenous_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_endogenous_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

                if self.args.check_data == True and self.args.batch_size == 1:
                    if test_data.scale and self.args.inverse:
                        shape_endx = seq_endogenous_x.shape
                        shape_exox = seq_exogenous_x.shape
                        seq_endogenous_x = seq_endogenous_x.detach().cpu().numpy()
                        seq_exogenous_x = seq_exogenous_x.detach().cpu().numpy()
                        seq_endogenous_x_inversed = test_data.inverse_endogenous_transform(seq_endogenous_x.squeeze(0)).reshape(shape_endx)
                        seq_endogenous_y_inversed = seq_endogenous_y
                        seq_exogenous_x_inversed = test_data.inverse_exogenous_transform(seq_exogenous_x.squeeze(0)).reshape(shape_exox)
                        print(f"batch_size: {i}, train_others: {seq_exogenous_x_inversed} train_CO2: {seq_endogenous_x_inversed} true_CO2 {seq_endogenous_y_inversed}")
                        print(f"pred_CO2:", pred)

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('{}: mse:{}, mae:{}'.format(get_now(),mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

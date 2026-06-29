import copy
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import time
import torch
import torch.utils.data as data
import os
import numpy as np
import progressbar as PB
from sklearn.metrics import r2_score
from torch.utils.tensorboard import SummaryWriter as SW
from io import StringIO
import sys


def R2(y_true, y_pred):
    return r2_score(y_true, y_pred)


def ME(y_true, y_pred):
    return np.mean(y_true-y_pred)


def SD(y_true, y_pred):
    return np.std(y_true-y_pred)


widgets = [
    PB.Bar(),
    PB.Counter(),
    ' ',
    PB.Percentage(),
    ' ',
    PB.DynamicMessage('Batch_BP_Loss'),
    ' ',
    PB.ETA()
]


def pearson_correlation_loss(x, y):
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    numerator = torch.sum((x - x_mean) * (y - y_mean))
    denominator = torch.sqrt(torch.sum((x - x_mean)**2) * torch.sum((y - y_mean)**2) + 1e-8)
    r = numerator / denominator
    return 1.0 + r

class Model_Trainer:
    def __init__(
        self,
        model,
        criterion_BP,
        optimizer_BP,
        device,
        settings_yml,
        batch_size=32,
        num_epochs=100,
        save_states=False,
        save_final=False,
        target_label='BP',
        ptt_weight=0.05,
        model_id_suffix='',
        epoch_eval_names=None,
        show_batch_progress=False,
        print_model_info=False,
        scheduler_type='warm_restarts',
        use_himc=True,
    ):

        self.Model_Running = model.to(device)
        self.Model_BestTest = []
        self.BP_Loss_Fun = criterion_BP
        self.Optimizer_BP = optimizer_BP
        self.Num_Epoch = num_epochs
        self.Train_Batchsize = batch_size
        self.Device = device
        self.Save_States = save_states
        self.Save_Final = save_final
        self.YMLSettings = settings_yml
        self.target_label = target_label
        self.PTT_Loss_Weight = ptt_weight
        self.Model_ID_Suffix = model_id_suffix
        self.Epoch_Eval_Names = epoch_eval_names
        self.Show_Batch_Progress = show_batch_progress
        self.Print_Model_Info = print_model_info
        self.Scheduler_Type = scheduler_type
        self.Use_HIMC = use_himc

    def Model_Info(self):
        model = self.Model_Running
        print('-' * 10)
        print('HD-Net Structure:')
        print(model)
        num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Trainable parameters: {}'.format(num))
        print('Settings')
        for item, setting in self.YMLSettings.items():
            print(item, ':', setting)
        print('-' * 10)

    def Set_Dataset(self, train_set, test_set=[]):
        self.Train_Set = train_set
        self.Test_Set_List = test_set

    def Evaluate_Model(self, pth_path):
        import torch.utils.data as data
        print(f"\n[Evaluating HD-Net from: {pth_path}]")


        self.Model_Running = torch.load(pth_path, map_location=self.Device)
        self.Model_Running.eval()

        Test_Names = []
        Test_List = []
        for name, testdata in self.Test_Set_List.items():
            Test_Names.append(name)
            Test_List.append(data.DataLoader(testdata, batch_size=128))

        for name, Test in zip(Test_Names, Test_List):
            Epoch_Test_Loss = []
            Epoch_Preds = []
            Epoch_Labels = []
            with PB.ProgressBar(max_value=len(Test)) as bar:
                for k, (inputs, static_feats, labels) in enumerate(Test):
                    Loss_Per_Batch, Outputs = self.Test_Batch(
                        inputs, static_feats, labels)
                    Epoch_Test_Loss.append(Loss_Per_Batch)
                    Epoch_Labels.append(labels.cpu().detach().numpy())
                    Epoch_Preds.append(Outputs.cpu().detach().numpy())
                    bar.update(k)

            Epoch_Labels = np.concatenate(Epoch_Labels, axis=0)
            Epoch_Preds = np.concatenate(Epoch_Preds, axis=0)

            Epoch_Test_Loss = self.BP_Loss_Fun(torch.from_numpy(Epoch_Labels), torch.from_numpy(Epoch_Preds))
            Epoch_Test_R2 = R2(Epoch_Labels, Epoch_Preds)
            Epoch_Test_ME = ME(Epoch_Labels, Epoch_Preds)
            Epoch_Test_SD = SD(Epoch_Labels, Epoch_Preds)

            import math
            import sklearn.metrics
            mae = sklearn.metrics.mean_absolute_error(Epoch_Labels, Epoch_Preds)
            print(f"Results for {name}:")
            print(f"Loss (MSE): {Epoch_Test_Loss.item():e}")
            print(f"R2 : {Epoch_Test_R2:.4f}")
            print(f"ME : {Epoch_Test_ME:.4f}")
            print(f"SD : {Epoch_Test_SD:.4f}")
            print(f"MAE: {mae:.4f}\n")

    def Train_Model(self):
        TimeID = datetime.now().strftime('%Y_%m%d_%H%M%S')
        suffix = f"_{self.Model_ID_Suffix}" if self.Model_ID_Suffix else ""
        ModelID = f"{self.target_label}{suffix}_{TimeID[-6:]}"
        os.makedirs(ModelID, exist_ok=True)

        Start_Epoch = 1
        batchcounter = 1
        batchrecordcounter = 1



        Writer = SW(os.path.join('TensorBoard', TimeID))
        print('ModelID: '+ModelID)
        if self.Print_Model_Info:
            self.Model_Info()
        else:
            num = sum(p.numel() for p in self.Model_Running.parameters() if p.requires_grad)
            print(f"Trainable parameters: {num}")
            print("Settings")
            for item, setting in self.YMLSettings.items():
                print(item, ':', setting)


        save_stdout = sys.stdout
        result = StringIO()
        sys.stdout = result
        print('ModelID: '+ModelID)
        self.Model_Info()
        sys.stdout = save_stdout
        Writer.add_text('Model', result.getvalue().replace('\n', '     \n'))



        Train = data.DataLoader(
            self.Train_Set, self.Train_Batchsize, shuffle=True, drop_last=False)
        Test_Names = []
        Test_List = []
        for name, testdata in self.Test_Set_List.items():
            Test_Names.append(name)
            Test_List.append(data.DataLoader(testdata, batch_size=128))

        Start_Time = time.time()

        Interrupt = False

        Train_Batch = self.Train_Batch

        scheduler = None
        early_stop_patience = 20
        patience_counter = 0
        best_val_mae = float('inf')
        best_epoch = Start_Epoch
        best_model_weights = copy.deepcopy(self.Model_Running.state_dict())
        history = {
            'Train_Loss': [],
            'Val_Loss': [],
            'Test_CalBased_Loss': [],
            'Test_CalFree_Loss': [],
            'Train_MAE': [],
            'Val_MAE': [],
            'Test_CalBased_MAE': [],
            'Test_CalFree_MAE': [],
        }

        warmup_epochs = 5
        initial_lrs = [group['lr'] for group in self.Optimizer_BP.param_groups]
        scheduler_epochs = max(1, self.Num_Epoch - warmup_epochs)
        if self.Scheduler_Type == 'warm_restarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.Optimizer_BP, T_0=20, T_mult=1, eta_min=1e-6
            )
        elif self.Scheduler_Type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.Optimizer_BP, T_max=scheduler_epochs, eta_min=1e-6
            )
        elif self.Scheduler_Type == 'none':
            scheduler = None
        else:
            raise ValueError(f"Unsupported scheduler_type: {self.Scheduler_Type}")

        for Epoch in range(Start_Epoch, Start_Epoch+self.Num_Epoch):
            try:

                if Epoch <= warmup_epochs:
                    for i, group in enumerate(self.Optimizer_BP.param_groups):
                        group['lr'] = initial_lrs[i] * (Epoch / warmup_epochs)


                Epoch_BP_Preds = []
                Epoch_BP_Labels = []
                Epoch_BP_Train_Loss = 0.0
                total_batches = 0

                num_chunks = getattr(self.Train_Set, 'chunk_files', [None])
                for chunk_idx in range(len(num_chunks)):
                    if hasattr(self.Train_Set, 'load_chunk'):
                        self.Train_Set.load_chunk(chunk_idx)
                        Train = data.DataLoader(self.Train_Set, self.Train_Batchsize, shuffle=True, drop_last=False)

                    def consume_train_batch(inputs, static_feats, BP_labels):
                        nonlocal batchcounter, batchrecordcounter
                        BP_loss, BP_Outputs = Train_Batch(inputs, static_feats, BP_labels)
                        Epoch_BP_Labels.append(BP_labels.cpu().detach().numpy())
                        Epoch_BP_Preds.append(BP_Outputs.cpu().detach().numpy())
                        batchcounter += 1
                        if not batchcounter % 100:
                            Writer.add_scalar('Batch_BP_Loss', BP_loss, batchrecordcounter)
                            batchrecordcounter += 1
                        return BP_loss

                    if self.Show_Batch_Progress:
                        k = 0
                        with PB.ProgressBar(widgets=widgets, max_value=len(Train)) as bar:
                            for inputs, static_feats, BP_labels in Train:
                                BP_loss = consume_train_batch(inputs, static_feats, BP_labels)
                                bar.update(k, Batch_BP_Loss=BP_loss)
                                k += 1
                    else:
                        for inputs, static_feats, BP_labels in Train:
                            consume_train_batch(inputs, static_feats, BP_labels)


                Epoch_BP_Labels = np.concatenate(Epoch_BP_Labels, axis=0)
                Epoch_BP_Preds = np.concatenate(Epoch_BP_Preds, axis=0)
                Epoch_Train_R2 = R2(Epoch_BP_Labels, Epoch_BP_Preds)
                Epoch_Train_ME = ME(Epoch_BP_Labels, Epoch_BP_Preds)
                Epoch_Train_SD = SD(Epoch_BP_Labels, Epoch_BP_Preds)
                Epoch_Train_MAE = mean_absolute_error(Epoch_BP_Labels, Epoch_BP_Preds)


                Epoch_BP_Train_Loss = self.BP_Loss_Fun(torch.from_numpy(
                    Epoch_BP_Labels), torch.from_numpy(Epoch_BP_Preds))


                if self.Save_States:
                    self.Save_Checkpoint(
                        ModelID, TimeID, Epoch, batchcounter, batchrecordcounter, savemodel=False)


                Writer_Loss_Dict = {'Train_BP': Epoch_BP_Train_Loss}
                Writer_R2_Dict = {'Train': Epoch_Train_R2}
                Writer_ME_Dict = {'Train': Epoch_Train_ME}
                Writer_SD_Dict = {'Train': Epoch_Train_SD}
                Writer_MAE_Dict = {'Train': Epoch_Train_MAE}




                for name, Test in zip(Test_Names, Test_List):
                    if self.Epoch_Eval_Names is not None and name not in self.Epoch_Eval_Names:
                        continue
                    Test_Name = name
                    Epoch_Test_Loss = []
                    Epoch_Preds = []
                    Epoch_Labels = []

                    for inputs, static_feats, labels in Test:
                        Loss_Per_Batch, Outputs = self.Test_Batch(
                            inputs, static_feats, labels)
                        Epoch_Test_Loss.append(Loss_Per_Batch)
                        Epoch_Labels.append(labels.cpu().detach().numpy())
                        Epoch_Preds.append(Outputs.cpu().detach().numpy())

                    Epoch_Labels = np.concatenate(Epoch_Labels, axis=0)
                    Epoch_Preds = np.concatenate(Epoch_Preds, axis=0)

                    Epoch_Test_Loss = self.BP_Loss_Fun(torch.from_numpy(
                        Epoch_Labels), torch.from_numpy(Epoch_Preds))
                    Epoch_Test_R2 = R2(Epoch_Labels, Epoch_Preds)
                    Epoch_Test_ME = ME(Epoch_Labels, Epoch_Preds)
                    Epoch_Test_SD = SD(Epoch_Labels, Epoch_Preds)
                    Epoch_Test_MAE = mean_absolute_error(Epoch_Labels, Epoch_Preds)

                    Writer_Loss_Dict.update({Test_Name: Epoch_Test_Loss})
                    Writer_R2_Dict.update({Test_Name: Epoch_Test_R2})
                    Writer_ME_Dict.update({Test_Name: Epoch_Test_ME})
                    Writer_SD_Dict.update({Test_Name: Epoch_Test_SD})
                    Writer_MAE_Dict.update({Test_Name: Epoch_Test_MAE})


                Writer.add_scalars('Loss', Writer_Loss_Dict, Epoch)
                Writer.add_scalars('R2', Writer_R2_Dict, Epoch)
                Writer.add_scalars('ME', Writer_ME_Dict, Epoch)
                Writer.add_scalars('SD', Writer_SD_Dict, Epoch)
                Writer.add_scalars('MAE', Writer_MAE_Dict, Epoch)


                history['Train_Loss'].append(Epoch_BP_Train_Loss.item() if torch.is_tensor(Epoch_BP_Train_Loss) else float(Epoch_BP_Train_Loss))
                history['Train_MAE'].append(float(Epoch_Train_MAE))
                if 'Val' in Writer_Loss_Dict:
                    history['Val_Loss'].append(Writer_Loss_Dict['Val'].item() if torch.is_tensor(Writer_Loss_Dict['Val']) else float(Writer_Loss_Dict['Val']))
                if 'Val' in Writer_MAE_Dict:
                    history['Val_MAE'].append(float(Writer_MAE_Dict['Val']))
                if 'Test_CalFree' in Writer_Loss_Dict:
                    history['Test_CalFree_Loss'].append(Writer_Loss_Dict['Test_CalFree'].item() if torch.is_tensor(Writer_Loss_Dict['Test_CalFree']) else float(Writer_Loss_Dict['Test_CalFree']))
                if 'Test_CalFree' in Writer_MAE_Dict:
                    history['Test_CalFree_MAE'].append(float(Writer_MAE_Dict['Test_CalFree']))
                if 'Test_CalBased' in Writer_Loss_Dict:
                    history['Test_CalBased_Loss'].append(Writer_Loss_Dict['Test_CalBased'].item() if torch.is_tensor(Writer_Loss_Dict['Test_CalBased']) else float(Writer_Loss_Dict['Test_CalBased']))
                if 'Test_CalBased' in Writer_MAE_Dict:
                    history['Test_CalBased_MAE'].append(float(Writer_MAE_Dict['Test_CalBased']))

                val_loss = Writer_Loss_Dict.get('Val', Epoch_BP_Train_Loss)
                val_mae = float(Writer_MAE_Dict.get('Val', Epoch_Train_MAE))


                if Epoch > warmup_epochs:
                    if self.Scheduler_Type == 'warm_restarts':
                        scheduler.step(Epoch - warmup_epochs)
                    elif scheduler is not None:
                        scheduler.step()

                lr_str = " | ".join([f"LR{i}={g['lr']:.2e}" for i, g in enumerate(self.Optimizer_BP.param_groups)])

                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_epoch = Epoch
                    patience_counter = 0
                    best_model_weights = copy.deepcopy(self.Model_Running.state_dict())
                    best_marker = " *best"
                else:
                    patience_counter += 1
                    best_marker = ""

                epoch_parts = [
                    f"Epoch {Epoch:03d}/{Start_Epoch+self.Num_Epoch-1}",
                    f"TrainLoss={float(Epoch_BP_Train_Loss):.4f}",
                    f"TrainMAE={Epoch_Train_MAE:.4f}",
                ]
                for metric_name in Test_Names:
                    if metric_name in Writer_MAE_Dict:
                        metric_loss = Writer_Loss_Dict[metric_name]
                        metric_loss_value = metric_loss.item() if torch.is_tensor(metric_loss) else float(metric_loss)
                        epoch_parts.append(f"{metric_name}Loss={metric_loss_value:.4f}")
                        epoch_parts.append(f"{metric_name}MAE={Writer_MAE_Dict[metric_name]:.4f}")
                epoch_parts.append(lr_str)
                epoch_parts.append(f"BestValMAE={best_val_mae:.4f}@{best_epoch}{best_marker}")
                print(" | ".join(epoch_parts))

                if patience_counter >= early_stop_patience:
                    print(f"Early stopping triggered at Epoch {Epoch}.")
                    break


            except KeyboardInterrupt:
                print('Earlystopped by interrupt at epoch {:d}'.format(Epoch))
                Interrupt = True
                break
        Writer.close()
        time_elapsed = time.time() - Start_Time
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        self.Model_Running.load_state_dict(best_model_weights)
        print(f"Best HD-Net selected by Val MAE: epoch {best_epoch}, Val MAE {best_val_mae:.4f}")


        self.Save_Checkpoint(ModelID, TimeID, Epoch,
                             batchcounter, batchrecordcounter, savemodel=True)


        log_file = open(os.path.join(ModelID, 'evaluation_log.txt'), 'w')
        print(f"Final Evaluation with Best HD-Net: epoch {best_epoch}, Val MAE {best_val_mae:.4f}", file=log_file)

        for name, Test in zip(Test_Names, Test_List):
            all_preds = []
            all_labels = []
            for inputs, static_feats, labels in Test:
                _, Outputs = self.Test_Batch(inputs, static_feats, labels)
                all_preds.append(Outputs.cpu().detach().numpy())
                all_labels.append(labels.cpu().detach().numpy())

            all_preds = np.concatenate(all_preds, axis=0).flatten()
            all_labels = np.concatenate(all_labels, axis=0).flatten()

            mae = mean_absolute_error(all_labels, all_preds)
            me = ME(all_labels, all_preds)
            std = SD(all_labels, all_preds)
            r2 = R2(all_labels, all_preds)

            res_str = f"Dataset: {name} | MAE: {mae:.4f} | ME: {me:.4f} | STD: {std:.4f} | R2: {r2:.4f}"
            print(res_str)
            print(res_str, file=log_file)

        log_file.close()

        if Interrupt:            raise KeyboardInterrupt


    def Train_Batch(self, inputs, static_feats, BP_labels):
        self.Model_Running.train()
        inputs = inputs.float().to(self.Device)
        static_feats = static_feats.float().to(self.Device)
        BP_labels = BP_labels.float().to(self.Device)

        self.Model_Running.zero_grad()
        BP_outputs, ptt_index = self.Model_Running(inputs, static_feats)
        loss_reg = self.BP_Loss_Fun(BP_outputs, BP_labels)
        if self.Use_HIMC and ptt_index is not None:
            loss_ptt = pearson_correlation_loss(ptt_index, BP_labels)
            BP_loss = loss_reg + self.PTT_Loss_Weight * loss_ptt
        else:
            BP_loss = loss_reg
        BP_loss_report = BP_loss.item()

        BP_loss.backward()
        self.Optimizer_BP.step()

        return BP_loss_report, BP_outputs

    def Test_Batch(self, inputs, static_feats, labels):

        self.Model_Running.eval()
        inputs = inputs.float().to(self.Device)
        static_feats = static_feats.float().to(self.Device)
        labels = labels.float().to(self.Device)
        with torch.no_grad():

            pred_center, _ = self.Model_Running(inputs, static_feats)

            inputs_left_2 = torch.roll(inputs, shifts=-2, dims=-1)
            pred_left_2, _  = self.Model_Running(inputs_left_2, static_feats)

            inputs_right_2 = torch.roll(inputs, shifts=2, dims=-1)
            pred_right_2, _ = self.Model_Running(inputs_right_2, static_feats)

            inputs_left_5 = torch.roll(inputs, shifts=-5, dims=-1)
            pred_left_5, _  = self.Model_Running(inputs_left_5, static_feats)

            inputs_right_5 = torch.roll(inputs, shifts=5, dims=-1)
            pred_right_5, _ = self.Model_Running(inputs_right_5, static_feats)

            BP_outputs = (pred_center + pred_left_2 + pred_right_2 + pred_left_5 + pred_right_5) / 5.0

            BP_outputs = (pred_center + pred_left_2 + pred_right_2 + pred_left_5 + pred_right_5) / 5.0


            loss = self.BP_Loss_Fun(BP_outputs, labels)
        return loss.item(), BP_outputs

    def Save_Checkpoint(self, modelID, timeID, epoch, batchcounter, batchrecordcounter, savemodel=False):

        foldername = modelID
        if not os.path.isdir(foldername):
            os.mkdir(foldername)
        torch.save({'model_id': modelID,
                    'time_id': timeID,
                    'model_state_dict': self.Model_Running.state_dict(),
                    'optimizer_state_dict': self.Optimizer_BP.state_dict(),
                    'epoch': epoch,
                    'batchcounter': batchcounter,
                    'batchrecordcounter': batchrecordcounter,
                    }, os.path.join(foldername, 'checkpoint_epoch_{}.pth'.format(epoch)))
        if savemodel:
            torch.save(self.Model_Running, os.path.join(
                foldername, 'trained_model.pth'))


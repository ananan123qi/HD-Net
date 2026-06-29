import torch
import torch.utils.data as data
import random
import numpy as np
from Model_Training.Model_Def.Trainer import Model_Trainer
from Model_Training.Model_Def import ResNet
from Model_Training.Model_Training import NPZDataset, Seed, Train_File, Test_CalBased_File, Test_CalFree_File
import os
import glob

if __name__ == '__main__':



    for target_label in ['SBP', 'DBP']:
        print(f"\n{'='*50}\nStarting TTA Evaluation for {target_label}\n{'='*50}\n")
        Seed(6)


        Test_CalBased_Data = NPZDataset(Test_CalBased_File, target_label)
        Test_CalFree_Data = NPZDataset(Test_CalFree_File, target_label)


        model = ResNet.DualBranch_ResNet()
        Seed(6)

        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        model.to(device)


        BP_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        Settings = {'BP_optimizer': 'test_only', 'trainer': 'test_only'}


        model_trainer = Model_Trainer(model, torch.nn.MSELoss(), BP_optimizer, device, Settings, batch_size=128, target_label=target_label)
        model_trainer.Set_Dataset(None, {'Test_CalBased': Test_CalBased_Data, 'Test_CalFree': Test_CalFree_Data})


        if target_label == 'SBP':
            latest_model = os.path.join('checkpoints', 'SBP', 'trained_model.pth')
        else:
            latest_model = os.path.join('checkpoints', 'DBP', 'trained_model.pth')

        if not os.path.exists(latest_model):
            print(f"No trained HD-Net found at {latest_model}")
            continue

        print(f"Found trained HD-Net: {latest_model}")


        print(f"Applying TTA Evaluation (rolling shifts = [-5, -2, 0, +2, +5])")
        model_trainer.Evaluate_Model(latest_model)


        del model, BP_optimizer, model_trainer, Test_CalBased_Data, Test_CalFree_Data
        torch.cuda.empty_cache()


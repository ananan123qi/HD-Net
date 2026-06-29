import torch
import torch.utils.data as data
import random
import numpy as np
from mat73 import loadmat
from Model_Training.Model_Def.Trainer import Model_Trainer
from Model_Training.Model_Def import ResNet
import glob
import os
import h5py
import argparse
import re

def Seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class Chunked_Dataset(data.Dataset):
    def __init__(self, folder_path, label):
        self.chunk_files = sorted(glob.glob(os.path.join(folder_path, '*.mat')))
        self.label = label
        self.current_idx = -1
        self.Input = None
        self.Label = None
        self.length = 0

        self.load_chunk(0)

    def load_chunk(self, chunk_idx):
        if self.current_idx != chunk_idx:
            Data = loadmat(self.chunk_files[chunk_idx])
            self.Input = Data['Subset']['Signals'][:, 0:2, :].astype(np.float32)
            self.Label = Data['Subset'][self.label].astype(np.float32)
            self.length = len(self.Input)
            self.current_idx = chunk_idx

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.Input[idx, :], self.Label[[idx]]

class NPZDataset(data.Dataset):
    def __init__(self, file_path, label):
        self.file_path = file_path
        self.label = label


        self.mean = np.array([61.09467, 162.4307, 60.825035, 22.917908], dtype=np.float32)
        self.std = np.array([15.100339, 9.641375, 11.65865, 3.437813], dtype=np.float32)


        cache_dir = self.file_path + '_mmap_cache'
        required_cache_files = [
            'Signals.npy',
            f'{label}.npy',
            'Age.npy',
            'Height.npy',
            'Weight.npy',
            'Gender.npy',
        ]
        if os.path.isdir(cache_dir) and all(os.path.exists(os.path.join(cache_dir, name)) for name in required_cache_files):
            self.Input = np.load(os.path.join(cache_dir, 'Signals.npy'), mmap_mode='r', allow_pickle=False)
            self.Label = np.load(os.path.join(cache_dir, f'{label}.npy'), mmap_mode='r', allow_pickle=False)
            self.Age = np.load(os.path.join(cache_dir, 'Age.npy'), mmap_mode='r', allow_pickle=False)
            self.Height = np.load(os.path.join(cache_dir, 'Height.npy'), mmap_mode='r', allow_pickle=False)
            self.Weight = np.load(os.path.join(cache_dir, 'Weight.npy'), mmap_mode='r', allow_pickle=False)
            self.Gender = np.load(os.path.join(cache_dir, 'Gender.npy'), mmap_mode='r', allow_pickle=False)
        else:

            data_dict = np.load(self.file_path)


            self.Input = data_dict['Signals']
            self.Label = data_dict[label].astype(np.float32)


            self.Age = data_dict['Age'].astype(np.float32)
            self.Height = data_dict['Height'].astype(np.float32)
            self.Weight = data_dict['Weight'].astype(np.float32)
            self.Gender = data_dict['Gender']

    def __len__(self):
        return len(self.Input)

    def __getitem__(self, idx):
        signal = np.array(self.Input[idx, 0:2, :], dtype=np.float32, copy=True)
        label_val = np.array(self.Label[idx], dtype=np.float32, copy=True).reshape(-1)[:1]

        age = float(np.asarray(self.Age[idx]).reshape(-1)[0])
        gender = 1.0 if self.Gender[idx] == 'M' else 0.0
        height = float(np.asarray(self.Height[idx]).reshape(-1)[0])
        weight = float(np.asarray(self.Weight[idx]).reshape(-1)[0])

        if np.isnan(height) or height <= 0:
            height = float(self.mean[1])

        if np.isnan(weight) or weight <= 0:
            weight = float(self.mean[2])

        if np.isnan(age) or age <= 0:
            age = float(self.mean[0])

        if np.isnan(gender):
            gender = 1.0

        height_in_meters = height / 100.0
        bmi = weight / (height_in_meters ** 2)

        continuous_feats = np.array([age, height, weight, bmi], dtype=np.float32)

        nan_mask = np.isnan(continuous_feats)
        continuous_feats[nan_mask] = self.mean[nan_mask]


        normalized_continuous = (continuous_feats - self.mean) / (self.std + 1e-8)

        final_static_feats = np.array([
            normalized_continuous[0],
            gender,
            normalized_continuous[1],
            normalized_continuous[2],
            normalized_continuous[3]
        ], dtype=np.float32)

        return torch.FloatTensor(signal), torch.FloatTensor(final_static_feats), torch.FloatTensor(label_val)

DEFAULT_DATA_FOLDER = os.path.join('data', 'hdnet')
Train_File = os.path.join(DEFAULT_DATA_FOLDER, 'train.npz')
Val_File = os.path.join(DEFAULT_DATA_FOLDER, 'val.npz')
Test_CalBased_File = os.path.join(DEFAULT_DATA_FOLDER, 'test_calbased.npz')
Test_CalFree_File = os.path.join(DEFAULT_DATA_FOLDER, 'test_calfree.npz')


def set_data_files(data_folder):
    global Train_File, Val_File, Test_CalBased_File, Test_CalFree_File
    Train_File = os.path.join(data_folder, 'train.npz')
    Val_File = os.path.join(data_folder, 'val.npz')
    Test_CalBased_File = os.path.join(data_folder, 'test_calbased.npz')
    Test_CalFree_File = os.path.join(data_folder, 'test_calfree.npz')

class L1MSELoss(torch.nn.Module):
    def __init__(self, l1_weight=0.7, mse_weight=0.3):
        super().__init__()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight

    def forward(self, pred, target):
        return (
            self.l1_weight * torch.nn.functional.l1_loss(pred, target)
            + self.mse_weight * torch.nn.functional.mse_loss(pred, target)
        )



def sanitize_tag(tag):
    tag = re.sub(r'[^A-Za-z0-9_.-]+', '_', tag.strip())
    return tag.strip('_')


def build_criterion(loss_name, beta, l1_weight=0.7, mse_weight=0.3):
    if loss_name == 'mse':
        return torch.nn.MSELoss()
    if loss_name == 'smoothl1':
        return torch.nn.SmoothL1Loss(beta=beta)
    if loss_name == 'l1_mse':
        return L1MSELoss(l1_weight=l1_weight, mse_weight=mse_weight)
    raise ValueError(f"Unsupported loss: {loss_name}")


def make_config_from_args(args, target_label):
    tag = args.tag
    if not tag:
        if args.loss == 'smoothl1':
            tag = f"{args.loss}_b{args.beta:g}_p{args.ptt_weight:g}_base{args.base_lr:g}_film{args.film_lr:g}"
        else:
            tag = f"{args.loss}_p{args.ptt_weight:g}_base{args.base_lr:g}_film{args.film_lr:g}"
    return {
        'target': target_label,
        'loss': args.loss,
        'beta': args.beta,
        'l1_weight': args.l1_weight,
        'mse_weight': args.mse_weight,
        'ptt_weight': args.ptt_weight,
        'base_lr': args.base_lr,
        'film_lr': args.film_lr,
        'weight_decay_base': args.weight_decay_base,
        'weight_decay_film': args.weight_decay_film,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'save_states': args.save_states,
        'epoch_eval_names': ['Val'] if args.final_test_only else None,
        'show_batch_progress': args.show_batch_progress,
        'print_model_info': args.print_model_info,
        'scheduler': args.scheduler,
        'tag': sanitize_tag(tag),
    }


def run_one_experiment(config):
    target_label = config['target']
    print(f"\n{'='*70}")
    print(f"Starting HD-Net training for {target_label} | tag={config['tag']}")
    print(
        f"loss={config['loss']} beta={config.get('beta')} "
        f"l1_weight={config.get('l1_weight')} mse_weight={config.get('mse_weight')} "
        f"ptt_weight={config['ptt_weight']} base_lr={config['base_lr']} "
        f"film_lr={config['film_lr']} scheduler={config['scheduler']}"
    )
    print(f"{'='*70}\n")
    Seed(6)

    Train_Data = NPZDataset(Train_File, target_label)
    Val_Data = NPZDataset(Val_File, target_label)
    Test_CalBased_Data = NPZDataset(Test_CalBased_File, target_label)
    Test_CalFree_Data = NPZDataset(Test_CalFree_File, target_label)
    print(
        f"{target_label} split sizes | "
        f"train: {len(Train_Data)} | val: {len(Val_Data)} | "
        f"test_calbased: {len(Test_CalBased_Data)} | test_calfree: {len(Test_CalFree_Data)}"
    )

    model = ResNet.DualBranch_ResNet()
    Seed(6)

    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    model.to(device)

    new_modules = ['film', 'lstm', 'static_encoder', 'early_mapping', 'late_mapping', 'alpha_early', 'alpha_late', 'fc_reg']
    film_params = []
    base_params = []
    for name, param in model.named_parameters():
        if any(new_mod in name for new_mod in new_modules):
            film_params.append(param)
        else:
            base_params.append(param)

    BP_optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': config['base_lr'], 'weight_decay': config['weight_decay_base']},
        {'params': film_params, 'lr': config['film_lr'], 'weight_decay': config['weight_decay_film']}
    ])
    optimizer_desc = (
        f"AdamW(base_lr={config['base_lr']}, film_lr={config['film_lr']}, "
        f"base_wd={config['weight_decay_base']}, film_wd={config['weight_decay_film']})"
    )
    criterion_BP = build_criterion(
        config['loss'],
        config['beta'],
        config.get('l1_weight', 0.7),
        config.get('mse_weight', 0.3),
    )

    Settings = {
        'model': 'HD-Net',
        'BP_optimizer': optimizer_desc,
        'criterion': (
            f"{config['loss']}(beta={config['beta']}, "
            f"l1_weight={config.get('l1_weight')}, mse_weight={config.get('mse_weight')})"
        ),
        'ptt_weight': config['ptt_weight'],
        'use_himc': True,
        'selection_metric': 'Val_MAE',
        'tag': config['tag'],
        'save_states': config['save_states'],
        'epoch_eval_names': config['epoch_eval_names'] if config['epoch_eval_names'] is not None else 'all',
        'show_batch_progress': config['show_batch_progress'],
        'print_model_info': config['print_model_info'],
        'scheduler': config['scheduler'],
    }

    model_trainer = Model_Trainer(
        model,
        criterion_BP,
        BP_optimizer,
        device,
        Settings,
        batch_size=config['batch_size'],
        num_epochs=config['num_epochs'],
        save_states=config['save_states'],
        save_final=True,
        target_label=target_label,
        ptt_weight=config['ptt_weight'],
        model_id_suffix=config['tag'],
        epoch_eval_names=config['epoch_eval_names'],
        show_batch_progress=config['show_batch_progress'],
        print_model_info=config['print_model_info'],
        scheduler_type=config['scheduler'],
        use_himc=True,
    )
    model_trainer.Set_Dataset(
        Train_Data,
        {
            'Val': Val_Data,
            'Test_CalBased': Test_CalBased_Data,
            'Test_CalFree': Test_CalFree_Data,
        },
    )
    model_trainer.Train_Model()

    del model, BP_optimizer, model_trainer, Train_Data, Val_Data, Test_CalBased_Data, Test_CalFree_Data
    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description='Train HD-Net on prepared NPZ splits.')
    parser.add_argument('--data-folder', default=DEFAULT_DATA_FOLDER, help='Folder containing prepared HD-Net NPZ split files.')
    parser.add_argument('--targets', nargs='+', choices=['SBP', 'DBP'], default=['SBP', 'DBP'])
    parser.add_argument('--loss', choices=['mse', 'smoothl1', 'l1_mse'], default='smoothl1')
    parser.add_argument('--beta', type=float, default=5.0)
    parser.add_argument('--l1-weight', type=float, default=0.7)
    parser.add_argument('--mse-weight', type=float, default=0.3)
    parser.add_argument('--ptt-weight', type=float, default=0.05)
    parser.add_argument('--base-lr', type=float, default=5e-4)
    parser.add_argument('--film-lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay-base', type=float, default=1e-3)
    parser.add_argument('--weight-decay-film', type=float, default=1e-4)
    parser.add_argument('--scheduler', choices=['warm_restarts', 'cosine', 'none'], default='warm_restarts')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--save-states', action='store_true', help='Save checkpoint_epoch_N.pth every epoch.')
    parser.add_argument('--show-batch-progress', action='store_true', help='Print batch progress bars.')
    parser.add_argument('--print-model-info', action='store_true', help='Print the full HD-Net architecture.')
    parser.add_argument(
        '--final-test-only',
        action='store_true',
        help='Evaluate only Val during training; run Test_CalBased/Test_CalFree only once after training.',
    )
    parser.add_argument('--tag', default='', help='Suffix added to the output HD-Net folder name.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    set_data_files(args.data_folder)
    configs = [make_config_from_args(args, target_label) for target_label in args.targets]
    for experiment_config in configs:
        run_one_experiment(experiment_config)



import torch
import torch.nn as nn
from torch import Tensor

def conv3x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv1d:
    return nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv1d:
    return nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample = None, norm_layer = nn.BatchNorm1d):
        super(BasicBlock,self).__init__()
        self.conv1 = conv3x1(in_channels, out_channels, stride=stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(out_channels, out_channels, stride=1)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        return self.relu(out)

class ResNet1DBranch(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2,2,2,2], zero_init_residual=False, norm_layer=nn.BatchNorm1d):
        super(ResNet1DBranch, self).__init__()
        self._norm_layer = norm_layer
        self.input_channels = 64
        self.conv1 = nn.Conv1d(1, self.input_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.input_channels != out_channels * block.expansion:
            downsample = nn.Sequential(conv1x1(self.input_channels, out_channels * block.expansion, stride), norm_layer(out_channels * block.expansion))
        layers = []
        layers.append(block(self.input_channels, out_channels, stride, downsample, norm_layer))
        self.input_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.input_channels, out_channels, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        feat_s = self.layer2(x)
        x = self.layer3(feat_s)
        feat_d = self.layer4(x)
        return feat_s, feat_d

class DualBranch_ResNet(nn.Module):
    def __init__(self, num_BP=1):
        super(DualBranch_ResNet, self).__init__()
        self.ecg_branch = ResNet1DBranch()
        self.ppg_branch = ResNet1DBranch()


        self.time_delay_conv = nn.Conv1d(in_channels=256, out_channels=1, kernel_size=7, padding=3)


        self.film_s_mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256)
        )


        self.film_d_mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024)
        )

        self.shallow_align_conv = nn.Conv1d(128, 512, kernel_size=1)
        self.reduce_conv = nn.Conv1d(1536, 512, kernel_size=1)


        self.lstm = nn.LSTM(input_size=512, hidden_size=128, batch_first=True)

        self.state_dim = 1 * 1 * 128

        self.static_encoder = nn.Sequential(
            nn.Linear(5, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        self.early_mapping = nn.Sequential(
            nn.Linear(64, self.state_dim * 2),
            nn.Tanh()
        )

        self.late_mapping = nn.Linear(64, 32)

        self.alpha_early = nn.Parameter(torch.tensor(0.001))
        self.alpha_late = nn.Parameter(torch.tensor(0.001))

        self.fc_reg = nn.Linear(1184, 1)
    def forward(self, x, x_static, return_mask=False):
        ecg = x[:, 0:1, :]
        ppg = x[:, 1:2, :]
        ecg_feat_s, ecg_feat_d = self.ecg_branch(ecg)
        ppg_feat_s, ppg_feat_d = self.ppg_branch(ppg)



        cat_feat = torch.cat([ecg_feat_s, ppg_feat_s], dim=1)


        raw_mask = self.time_delay_conv(cat_feat)
        M_t = torch.sigmoid(raw_mask)


        ecg_s_masked = (M_t * ecg_feat_s).mean(dim=2)


        film_s_params = self.film_s_mlp(ecg_s_masked)
        gamma_s, beta_s = film_s_params.chunk(2, dim=1)
        gamma_s = gamma_s.unsqueeze(2)
        beta_s  = beta_s.unsqueeze(2)


        ppg_s_out = (1 + gamma_s) * ppg_feat_s + beta_s



        ecg_d_global = ecg_feat_d.mean(dim=2)


        film_d_params = self.film_d_mlp(ecg_d_global)
        gamma_d, beta_d = film_d_params.chunk(2, dim=1)
        gamma_d = gamma_d.unsqueeze(2)
        beta_d  = beta_d.unsqueeze(2)


        ppg_d_out = (1 + gamma_d) * ppg_feat_d + beta_d



        ppg_s_aligned = self.shallow_align_conv(ppg_s_out)


        ppg_s_aligned = torch.nn.functional.adaptive_avg_pool1d(ppg_s_aligned, ppg_d_out.shape[2])


        fused = torch.cat([ppg_s_aligned, ppg_d_out, ecg_feat_d], dim=1)
        fused = self.reduce_conv(fused)


        cnn_mean = fused.mean(dim=2)
        cnn_max = fused.max(dim=2)[0]


        lstm_input = fused.transpose(1, 2)
        batch_size = lstm_input.size(0)

        static_hidden = self.static_encoder(x_static)

        early_features = self.early_mapping(static_hidden) * self.alpha_early
        h_0_flat, c_0_flat = torch.chunk(early_features, 2, dim=1)
        h_0 = h_0_flat.view(1, batch_size, 128).contiguous()
        c_0 = c_0_flat.view(1, batch_size, 128).contiguous()

        lstm_out, (hn, cn) = self.lstm(lstm_input, (h_0, c_0))
        lstm_final = hn[0]

        late_features = self.late_mapping(static_hidden) * self.alpha_late

        out = torch.cat((lstm_final, cnn_mean, cnn_max, late_features), dim=1)


        bp_values = self.fc_reg(out)

        ptt_index = M_t.mean(dim=2)

        if return_mask:
            attn_dict = {
                'M_t': M_t,
                'gamma_s': gamma_s,
                'beta_s': beta_s,
                'gamma_d': gamma_d,
                'beta_d': beta_d,
                'late_features': late_features
            }
            return bp_values, ptt_index, attn_dict
        return bp_values, ptt_index

def Resnet18_1D(**kwargs):pass

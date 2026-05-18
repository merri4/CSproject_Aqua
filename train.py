import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

import wandb
from tqdm import tqdm
import argparse
import os


# Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, data, lookback=48, horizon=24):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.lookback = lookback
        self.horizon = horizon
        
    def __len__(self):
        return len(self.data) - self.lookback - self.horizon + 1
        
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.lookback]
        y = self.data[idx + self.lookback : idx + self.lookback + self.horizon]
        return x, y

# Model
class AutoregressiveLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(AutoregressiveLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        pred = self.fc(out)
        return pred

class AutoregressiveTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, nhead=4, max_len=500, dropout=0.1):
        super(AutoregressiveTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = nn.Embedding(max_len, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # 선형 사영 및 위치 인코딩 추가
        out = self.input_proj(x)
        pos = torch.arange(0, seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        out = out + self.pos_encoder(pos)
        
        # 어텐션 연산 시 미래 시점을 보지 못하도록 Causal Mask 생성
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        
        out = self.transformer_encoder(out, mask=mask)
        pred = self.fc(out)
        return pred

def parse_arguments() :

    parser = argparse.ArgumentParser(description='Argparse')

    # Directories
    parser.add_argument('--data_path', type=str, default='./pond2_resampled_15min.csv')
    parser.add_argument('--model_type', type=str, default='lstm', choices=['lstm', 'transformer'], help='Backbone model type')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save graphs and model checkpoints')

    # Preprocessing Hyperparameters
    parser.add_argument('--lookback', type=int, default=48) # 과거 12시간 (48 * 15분)
    parser.add_argument('--horizon', type=int, default=24)  # 미래 6시간  (24 * 15분)
    
    # Model Hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_blocks', type=int, default=2)

    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)

    args = parser.parse_args()
    return args


def train(args) :

    # 출력 디렉터리 사전 생성
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # =================================
    # Data load, Preprocesssing
    # =================================
    df = pd.read_csv(args.data_path)

    # 분포가 큰 암모니아는 로그 스케일 적용하여 안정화
    df['Ammonia(g/ml)'] = np.log1p(df['Ammonia(g/ml)'])
    features = ['Temperature (C)', 'Turbidity (NTU)', 'Dissolved Oxygen(g/ml)', 'PH', 'Ammonia(g/ml)', 'Nitrate(g/ml)']
    data = df[features].values

    # Data split (train:val 8:2)
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # Add scaler (Train에만 fit)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)

    # Overlapping Sliding Window Dataset 
    train_dataset = TimeSeriesDataset(train_scaled, args.lookback, args.horizon)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    val_dataset = TimeSeriesDataset(val_scaled, args.lookback, args.horizon)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)


    # =================================
    # Model loading
    # =================================
    
    # 아키텍처 선택
    if args.model_type == 'lstm':
        model = AutoregressiveLSTM(input_dim=len(features), hidden_dim=args.hidden_dim, num_layers=args.num_blocks).to(device)
    elif args.model_type == 'transformer':
        model = AutoregressiveTransformer(input_dim=len(features), hidden_dim=args.hidden_dim, num_layers=args.num_blocks).to(device)
    
    criterion = nn.L1Loss()  # MAE
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    logger = wandb.init(
        entity="kjhwang97",
        project="C&S_Aqua",
        dir="/home/merri4/wandb",
        config={

            # dataset Hyperparameter
            "dataset" : args.data_path,
            "lookback" : args.lookback,
            "horizon" : args.horizon,
            
            "hidden_dim" : args.hidden_dim,
            "num_blocks" : args.num_blocks,
            
            "batch_size" : args.batch_size,
            "lr" : args.lr,
            "epochs" : args.epochs,
        },
    )

    # =================================
    # Training
    # =================================

    for epoch in tqdm(range(args.epochs), desc="Training") :
        
        epoch_loss = 0
        model.train()
        for batch_x, batch_y in train_loader:

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Teacher Forcing: 입력과 정답(마지막 시점 제외)을 연결하여 모델에 입력
            full_input = torch.cat([batch_x, batch_y[:, :-1, :]], dim=1)
            
            optimizer.zero_grad()
            pred = model(full_input)
            
            # 예측 구간인 Horizon 영역에 대해서만 Loss 산출
            pred_horizon = pred[:, args.lookback-1:, :]
            loss = criterion(pred_horizon, batch_y)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # =================================
        # Validation
        # =================================
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:

                batch_x = batch_x.to(device)
                current_input = batch_x.clone()
                rollout_preds = []
                
                # 24스텝 동안 Autoregressive하게 루프를 돌며 미래 예측
                for step in range(args.horizon):
                    pred = model(current_input)
                    next_pred = pred[:, -1:, :]  # 최신 시점의 예측값
                    rollout_preds.append(next_pred)
                    
                    # 입력 창(Window)을 한 칸 밀고 새로운 예측값을 추가
                    current_input = torch.cat([current_input[:, 1:, :], next_pred], dim=1)
                    
                rollout_preds = torch.cat(rollout_preds, dim=1)
                all_preds.append(rollout_preds.cpu().numpy())
                all_targets.append(batch_y.numpy())

        # 원본 스케일 복원 및 성능 메트릭 평가
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        N, H, F = all_preds.shape

        preds_orig = scaler.inverse_transform(all_preds.reshape(-1, F)).reshape(N, H, F)
        targets_orig = scaler.inverse_transform(all_targets.reshape(-1, F)).reshape(N, H, F)

        # 암모니아 로그 역변환
        preds_orig[:, :, 4] = np.expm1(preds_orig[:, :, 4])
        targets_orig[:, :, 4] = np.expm1(targets_orig[:, :, 4])

        # wandb에 로깅
        metrics = {"train_loss" : epoch_loss/len(train_loader)}
        for i, col_name in enumerate(features):
            mae = mean_absolute_error(targets_orig[:, :, i].flatten(), preds_orig[:, :, i].flatten())
            rmse = np.sqrt(mean_squared_error(targets_orig[:, :, i].flatten(), preds_orig[:, :, i].flatten()))
            metrics[f"{col_name}_MAE"] = mae
            metrics[f"{col_name}_RMSE"] = rmse
        logger.log(metrics)

        # 마지막 에포크일 때 시각화 그래프 파일 저장 및 가중치 저장 실행
        if epoch == args.epochs - 1 and args.output_dir:
            # 6시간 자가회귀 롤아웃 시각화 플롯 플로팅 (검증 세트 내 임의 샘플 고정)
            sample_idx = min(100, len(targets_orig) - 1)
            fig, axes = plt.subplots(3, 2, figsize=(16, 12))
            axes = axes.flatten()
            time_steps = np.arange(1, args.horizon + 1) * 15 # 분 단위 변환

            for f_idx, col_name in enumerate(features):
                axes[f_idx].plot(time_steps, targets_orig[sample_idx, :, f_idx], 'g-o', label='Actual (Ground Truth)')
                axes[f_idx].plot(time_steps, preds_orig[sample_idx, :, f_idx], 'r--s', label='Autoregressive Rollout')
                axes[f_idx].set_title(f'{col_name} Forecast')
                axes[f_idx].set_xlabel('Minutes Ahead')
                axes[f_idx].set_ylabel(col_name)
                axes[f_idx].legend()
                axes[f_idx].grid(True, alpha=0.3)

            plt.suptitle(f'6-Hour Autoregressive Rollout Validation ({args.model_type.upper()})', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # 그래프 저장
            graph_save_path = os.path.join(args.output_dir, f"rollout_forecast_{args.model_type}.png")
            plt.savefig(graph_save_path)
            plt.close()
            print(f"\n[Saved] 6-hour forecast plot saved to: {graph_save_path}")

    # 최종 가중치 모델 파일 저장 (.pth)
    if args.output_dir:
        model_save_path = os.path.join(args.output_dir, f"{args.model_type}_model.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"[Saved] Final {args.model_type.upper()} model saved to: {model_save_path}")

if __name__ == "__main__":
    args = parse_arguments()
    train(args)
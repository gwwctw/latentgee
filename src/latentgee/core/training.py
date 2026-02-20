
import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from .losses import ziln_nll

def train_vae(model, data_tensor, epochs=50, lr=1e-3,  device="cpu", batch_size=256):
    # 0) 타입 체크
    # Ensure model is an instance of nn.Module and data_tensor is a torch.Tensor before using .to(device)
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module instance")
    if not isinstance(data_tensor, torch.Tensor):
        raise TypeError("data_tensor must be a torch.Tensor instance")
    
    # 1) 디바이스/AMP 설정
    device_str = device if isinstance(device, str) else device.type
    model = model.to(device_str)
    use_amp = (device_str == "cuda")  # GPU에서만 AMP 사용
    # 2) DataLoader는 **CPU 텐서**로 만든다 (pin_memory=True 사용 가능)
    # 데이터가 CPU에서 GPU로 오기까지의 I/O 병목을 없애 GPU 대기시간 제거
    
    num_workers = min(8, max(2, os.cpu_count() - 2))   # CPU코어-2, 최대 8 정도
    data_cpu = data_tensor.detach().to("cpu")
    ds = TensorDataset(data_cpu)  # 항상 CPU 텐서                      # ★ 중요: data_cpu 사용
    dl = DataLoader(ds, 
                    batch_size=batch_size, 
                    shuffle=True,
                    num_workers=num_workers,
                    prefetch_factor=4,
                    persistent_workers=True,
                    pin_memory=(device_str == "cuda"),
                    drop_last=True)  # 모든 스텝에서 입력 shape를 동일하게 유지하고 싶다면

    # 3) 옵티마이저/스케일러
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler(device="cuda", enabled=use_amp)

    last_loss = None

    for ep in range(epochs):
        model.train()
        for (xb,) in dl:
            # 4) 배치를 GPU로 전송 (pin_memory=True면 non_blocking=True로 이득)
            xb = xb.to(device_str, non_blocking=use_amp)

            with autocast(device_type="cuda", enabled=use_amp):
                (pi, mu_x, logσ_x), mu_z, logvar_z, _ = model(xb)
                recon_nll = ziln_nll(xb, pi, mu_x, logσ_x)  # 사용자 정의 NLL
                kl = -0.5 * torch.mean(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
                loss: torch.Tensor = recon_nll + kl

            if torch.isnan(loss):
                # 튜닝에서 이 trial을 최악으로 처리하고 끝내려면 큰 값 반환
                return float("inf")

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            last_loss = loss

        # if (ep + 1) % 10 == 0:
        #     print(f"[{ep+1}/{epochs}] ZILN-NLL {recon_nll.item():.4f}  KL {kl.item():.4f}")

    # 5) 마지막 loss를 float로 반환
    if last_loss is None:
        # 데이터가 비어있지 않다면 여긴 오면 안 됨
        return float("inf")
    return float(last_loss.detach().cpu().item())



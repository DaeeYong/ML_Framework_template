# scripts/inference.py
import os
import yaml
import torch
import numpy as np
from src.models.simple_net import SimpleNet

def main():
    # 설정 로드
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    final_checkpoint_path = config["paths"]["final_checkpoint"]

    input_dim = config["model"]["input_dim"]
    hidden_dim = config["model"]["hidden_dim"]
    num_classes = config["model"]["num_classes"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 최종 모델 로드
    model = SimpleNet(input_dim, hidden_dim, num_classes).to(device)
    model.load_state_dict(torch.load(final_checkpoint_path))
    model.eval()

    # 간단히 임의의 데이터로 추론
    x = np.random.randn(1, input_dim).astype(np.float32)
    x_tensor = torch.tensor(x, device=device)
    outputs = model(x_tensor)
    _, predicted = torch.max(outputs, 1)
    print(f"Inference result: {predicted.item()}")

if __name__ == "__main__":
    main()
# scripts/eval.py
import os
import yaml
import torch
from torch.utils.data import DataLoader
from src.data.dataset import FakeDataset
from src.models.simple_net import SimpleNet
from src.evaluation.evaluate import evaluate

def main():
    # 설정 로드
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    data_path = config["paths"]["data_path"]
    experiment_dir = config["paths"]["experiment_dir"]
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")

    # 베스트 모델 로드 (예시)
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")

    input_dim = config["model"]["input_dim"]
    hidden_dim = config["model"]["hidden_dim"]
    num_classes = config["model"]["num_classes"]

    dataset = FakeDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNet(input_dim, hidden_dim, num_classes).to(device)
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    val_loss, val_acc = evaluate(model, dataloader, device)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    main()
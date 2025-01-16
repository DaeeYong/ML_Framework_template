# scripts/train.py
import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.dataset import FakeDataset
from src.models.simple_net import SimpleNet
from src.training.train_loop import train_one_epoch
from src.evaluation.evaluate import evaluate

def main():
    # 1) 설정 로드
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    data_path = config["paths"]["data_path"]
    experiment_dir = config["paths"]["experiment_dir"]
    os.makedirs(experiment_dir, exist_ok=True)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    batch_size = config["training"]["batch_size"]
    lr = config["training"]["lr"]
    epochs = config["training"]["epochs"]

    input_dim = config["model"]["input_dim"]
    hidden_dim = config["model"]["hidden_dim"]
    num_classes = config["model"]["num_classes"]

    # 2) 데이터셋 & 데이터로더
    dataset = FakeDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3) 모델 준비
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNet(input_dim, hidden_dim, num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0

    # 4) 학습 루프
    for epoch in range(1, epochs+1):
        train_loss, train_acc = train_one_epoch(model, dataloader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, dataloader, device)  # 여기선 같은 데이터로 평가(데모용)

        print(f"[Epoch {epoch}/{epochs}] train_loss: {train_loss:.4f}, "
              f"train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

        # 체크포인트 저장
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }, checkpoint_path)

        # 베스트 모델 갱신
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(model.state_dict(), best_path)

    print("Training finished.")

if __name__ == "__main__":
    main()
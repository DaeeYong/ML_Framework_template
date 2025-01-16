import os
import shutil

experiment_dir = "experiments/experiment_1"
best_model_src = os.path.join(experiment_dir, "checkpoints", "best_model.pt")
final_model_dst = "checkpoints/final_model.pth"

shutil.copyfile(best_model_src, final_model_dst)
print("Final model saved to checkpoints/final_model.pth")
from pathlib import Path
import os
import argparse
import json
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def load_jsonl(file_path):
  with open(file_path, "r") as f:
    return [json.loads(line) for line in f]

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", type=str, default="pubmedqa")
  parser.add_argument("--model", type=str, default="microsoft/Phi-3.5-mini-instruct")
  return parser.parse_args()

##############################################
project_root = Path("ROOT_DIR")
#############################################

def evaluate_line_graph(model, task):
  #################################################################
  ploted_dir = project_root / "PROT_DIR"
  ploted_dir.mkdir(parents=True, exist_ok=True)
  inferenced_dir = project_root / "INFERENCED_OUTPUT_DIR"
  base_model_suffix = model.split("/")[-1]
  base_line_file = inferenced_dir / f"{task}.{base_model_suffix}.inferenced.jsonl"
  ##################################################################

  file_paths = []
  file_paths.append(base_line_file)
  rag_models = ["ncbi/MedCPT-Query-Encoder", "facebook/contriever", "allenai/specter", "BM25"]
  rag_dbs = ["statpearls", "textbooks"]
  top_k = [1, 3]
  length = [256, 512]

  for rag_model in rag_models:
    for rag_db in rag_dbs:
      for k in top_k:
        for l in length:
          rag_model_suffix = rag_model.split("/")[-1]
          file_path = inferenced_dir / f"{task}.{rag_db}.{rag_model_suffix}.{l}length.{base_model_suffix}.top{k}.inferenced.jsonl"
          file_paths.append(file_path)

  accuracy, actuals, predicted_probs = [], [], []
  plt.figure(figsize=(16, 9))
  plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration", linewidth=3)
  save_filename = f"{task}.{base_model_suffix}.calibration_curve.png"

  for file_path in file_paths:
    data = load_jsonl(file_path)
    for line in data:
      accuracy.append(1 if line["is_correct"] == True else 0)
      actuals.append(1 if line["is_correct"] == True else 0)
      predicted_probs.append(line["best_choice"]["best_probability"])

    prob_true, prob_pred = calibration_curve(actuals, predicted_probs, n_bins=20)
    model_label = file_path.stem
    plt.plot(prob_pred, prob_true, marker="o", label=model_label, linewidth=3)
  plt.fill_between([0, 1], [0, 1], color="red", alpha=0.2)
  plt.fill_between([0, 1], [0, 1], [1, 1], color="skyblue", alpha=0.2)
  plt.xticks(fontsize=15)
  plt.yticks(fontsize=15)
  plt.grid()
  plt.tight_layout()
  plt.savefig(ploted_dir / save_filename)
  print(f"Done! Evaluation results are saved to {ploted_dir / save_filename}")

if __name__ == "__main__":
  args = parse_args()
  task = args.task
  model = args.model
  print(f"task: {task}, model: {model}")

  evaluate_line_graph(model, task)

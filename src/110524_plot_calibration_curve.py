"""
モデルごとのcalibration curveのパターンを全て1枚のグラフにプロットする
"""

from pathlib import Path
import os
import argparse
import json
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def load_jsonl(file_path):
  with open(file_path, "r") as f:
    return [json.loads(line) for line in f]


project_root = Path("/cl/home2/shintaro/rag-notebook")


def evaluate_line_graph(model, task):
  ploted_dir = project_root / "shintaro/plotted"
  ploted_dir.mkdir(parents=True, exist_ok=True)
  inferenced_dir = project_root / "make_datastore_py310/data/inferenced"
  base_model_suffix = model.split("/")[-1]
  base_line_file = inferenced_dir / f"{task}.{base_model_suffix}.inferenced.jsonl"

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
  # 線を太く
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
  # 背景を水色に
  plt.fill_between([0, 1], [0, 1], color="red", alpha=0.2)
  # 逆に上を赤く
  plt.fill_between([0, 1], [0, 1], [1, 1], color="skyblue", alpha=0.2)
  # plotの値をお大きくする

  plt.xticks(fontsize=15)
  plt.yticks(fontsize=15)

  # plt.xlabel("Mean Predicted Probability")
  # plt.ylabel("Fraction of Positives")
  # plt.title("Calibration Curve")
  # plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
  plt.grid()
  plt.tight_layout()
  plt.savefig(ploted_dir / save_filename)
  print(f"Done! Evaluation results are saved to {ploted_dir / save_filename}")


# task: pubmedqa, medmcqa, medqa, mmlu
# models: microsoft/Phi-3.5-mini-instruct, axiong/PMC_LLaMA_13B, meta-llama/Llama-3.1-70B, epfl-llm/meditron-70b
def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", type=str, default="pubmedqa")
  parser.add_argument("--model", type=str, default="microsoft/Phi-3.5-mini-instruct")
  return parser.parse_args()


if __name__ == "__main__":
  args = parse_args()
  task = args.task
  model = args.model
  print(f"task: {task}, model: {model}")

  evaluate_line_graph(model, task)

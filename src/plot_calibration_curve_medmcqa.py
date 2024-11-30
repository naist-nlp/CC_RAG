from pathlib import Path
import os
import argparse
import json
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=str, default="microsoft/Phi-3.5-mini-instruct")
  return parser.parse_args()

def load_jsonl(file_path):
  with open(file_path, "r") as f:
    return [json.loads(line) for line in f]

################################
project_root = Path("ROOT_PATH")
################################

def evaluate_line_graph(model):
  ####################################################
  ploted_dir = project_root / "YOUR_PLOT_DIR"
  ploted_dir.mkdir(parents=True, exist_ok=True)
  inferenced_dir = project_root / "YOUR_INFERENCED_DIR"
  manipulated_inferenced_dir = project_root / "YOUR_MANIPULATED_INFERENCED_DIR"
  base_model_suffix = model.split("/")[-1]
  base_line_file = inferenced_dir / f"medmcqa.{base_model_suffix}.inferenced.jsonl"
  ####################################################

  file_paths = []
  file_paths.append(base_line_file)
  prompt_patterns = ["1", "2", "3"]
  for prompt_pattern in prompt_patterns:
    ans1_filename = manipulated_inferenced_dir / f"ans1.prompt{prompt_pattern}.{base_model_suffix}.manipulated.jsonl"
    ans1_other2_filename = manipulated_inferenced_dir / f"ans1.other2.prompt{prompt_pattern}.{base_model_suffix}.manipulated.jsonl"
    other3_filename = manipulated_inferenced_dir / f"other3.prompt{prompt_pattern}.{base_model_suffix}.manipulated.jsonl"
    file_paths.append(ans1_filename)
    file_paths.append(ans1_other2_filename)
    file_paths.append(other3_filename)

  accuracy, actuals, predicted_probs = [], [], []
  plt.figure(figsize=(16, 9))
  plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
  save_filename = f"{base_model_suffix}.calibration_curve.png"

  for file_path in file_paths:
    try:
      data = load_jsonl(file_path)
    except:
      data = [{"is_correct": False, "best_choice": {"best_probability": 0}}]
    for line in data:
      accuracy.append(1 if line["is_correct"] == True else 0)
      actuals.append(1 if line["is_correct"] == True else 0)
      predicted_probs.append(line["best_choice"]["best_probability"])

    prob_true, prob_pred = calibration_curve(actuals, predicted_probs, n_bins=20)
    model_label = file_path.stem
    plt.plot(prob_pred, prob_true, marker="o", label=model_label)
  plt.xlabel("Mean Predicted Probability")
  plt.ylabel("Fraction of Positives")
  plt.title("Calibration Curve")
  plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
  plt.grid()
  plt.tight_layout()
  plt.savefig(ploted_dir / save_filename)
  print(f"Done! Evaluation results are saved to {ploted_dir / save_filename}")

if __name__ == "__main__":
  args = parse_args()
  model = args.model
  evaluate_line_graph(model)

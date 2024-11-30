from pathlib import Path
import json
import numpy as np
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import argparse

def load_jsonl(file_path):
  with open(file_path, "r") as f:
    return [json.loads(line) for line in f]


def save_jsonl(file_path, data):
  with open(file_path, "w") as f:
    for line in data:
      f.write(json.dumps(line) + "\n")


def process_data(data):
  probabilities = []
  correct_labels = []
  for entry in data:
    best_probability = entry["best_choice"]["best_probability"]
    is_correct = entry["is_correct"]

    probabilities.append(best_probability)
    correct_labels.append(is_correct)
  return probabilities, correct_labels


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--base_model_name", type=str, required=True)
  parser.add_argument("--task", type=str, required=True)
  return parser.parse_args()


def calculate_ACE(probabilities, correct_labels, num_ranges=10):
  # Sort predictions and labels by probability
  sorted_indices = np.argsort(probabilities)
  sorted_probs = np.array(probabilities)[sorted_indices]
  sorted_labels = np.array(correct_labels)[sorted_indices]
  N = len(probabilities)
  range_size = N // num_ranges
  ace_sum = 0.0
  for r in range(num_ranges):
    start_idx = r * range_size
    end_idx = (
        r +
        1) * range_size if r < num_ranges - 1 else N
    range_probs = sorted_probs[start_idx:end_idx]
    range_labels = sorted_labels[start_idx:end_idx]
    confidence = np.mean(range_probs)
    accuracy = np.mean(range_labels)
    ace_sum += abs(accuracy - confidence)
  ace = ace_sum / num_ranges
  return ace


def calculating_ECE(accuracy_by_bin, total_counts_by_bin, bin_centers):
  accuracy_by_bin = np.array(accuracy_by_bin)
  total_counts_by_bin = np.array(total_counts_by_bin)
  bin_centers = np.array(bin_centers)
  ece = np.sum(total_counts_by_bin * np.abs(accuracy_by_bin - bin_centers))
  ece /= np.sum(total_counts_by_bin)
  return ece


def calculating_MCE(accuracy_by_bin, bin_centers):
  accuracy_by_bin = np.array(accuracy_by_bin)
  bin_centers = np.array(bin_centers)
  mce = max(abs(accuracy_by_bin - bin_centers))
  return mce


def calculating_RMSCE(accuracy_by_bin, bin_centers):
  accuracy_by_bin = np.array(accuracy_by_bin)
  bin_centers = np.array(bin_centers)
  rmsce = np.sqrt(np.mean(np.square(accuracy_by_bin - bin_centers)))
  return rmsce


def evaluate_line_graph(file_path, save_filename):
  evaluated_data = []
  actuals = []
  predicted_probs = []
  data = load_jsonl(file_path)
  accuracy = []
  for line in data:
    accuracy.append(1 if line["is_correct"] else 0)
  eps = 1e-15
  accuracy = sum(accuracy) / (len(accuracy) + eps)

  for line in data:
    actuals.append(1 if line["is_correct"] else 0)
    predicted_probs.append(line["best_choice"]["best_probability"])

  eps = 1e-15
  accuracy = sum(actuals) / (len(actuals) + eps)
  prob_true, prob_pred = calibration_curve(actuals, predicted_probs, n_bins=20)
  plt.figure(figsize=(8, 6))
  plt.plot(prob_pred, prob_true, marker="o", label="Calibration Curve")
  plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
  plt.xlabel("Mean Predicted Probability")
  plt.ylabel("Fraction of Positives")
  plt.title("Calibration Curve")
  plt.legend()
  plt.grid()

  plt.savefig(save_filename)
  print(f"Data length: {len(data)}")
  print(f"Correct answers: {sum(actuals)}")
  print(f"Accuracy: {accuracy:.4f}")

  evaluated_data.append({
      "data_length": len(data),
      "correct_answers": sum(actuals),
      "accuracy": accuracy,
      "calibration_curve": str(save_filename),
  })
  print(f"Done! Evaluation results are saved to {save_filename}")
  return evaluated_data


def evaluate_bar_graph(file_path, num_bins, save_filename):
  data = load_jsonl(file_path)
  probabilities, correct_labels = process_data(data)
  bins = np.linspace(0, 1, num_bins + 1)
  bin_ranges = [(bin_start, bin_end) for bin_start, bin_end in zip(bins[:-1], bins[1:])]
  bin_centers = [(bin_start + bin_end) / 2 for bin_start, bin_end in bin_ranges]
  accuracy_by_bin = []
  correct_counts_by_bin = []
  total_counts_by_bin = []

  for bin_start, bin_end in bin_ranges:
    bin_indices = [i for i, p in enumerate(probabilities) if bin_start <= p < bin_end]
    total_count = len(bin_indices)
    true_count = sum([correct_labels[i] for i in bin_indices])

    if total_count > 0:
      accuracy = true_count / total_count
    else:
      accuracy = 0

    accuracy_by_bin.append(accuracy)
    correct_counts_by_bin.append(true_count)
    total_counts_by_bin.append(total_count)
  ece = (
      calculating_ECE(accuracy_by_bin, total_counts_by_bin, bin_centers)
      if sum(total_counts_by_bin) > 0 else 0)
  mce = calculating_MCE(accuracy_by_bin, bin_centers)
  rmsce = calculating_RMSCE(accuracy_by_bin, bin_centers)
  num_ranges = 10
  ace = calculate_ACE(probabilities, correct_labels, num_ranges)

  print(f"==========\nECE: {ece:.4f}\nMCE: {mce:.4f}\nRMSCE: {rmsce:.4f}\nACE: {ace:.4f}==========")

  plt.figure(figsize=(20, 10))
  plt.bar(bin_centers, accuracy_by_bin, width=(bins[1] - bins[0]), alpha=0.7, edgecolor="black")
  eps = 1e-15
  overall_accuracy = sum(correct_labels) / (len(correct_labels) + eps)
  plt.axhline(y=overall_accuracy, color="r", linestyle="-", label="Overall Accuracy")
  plt.text(0.95, overall_accuracy + 0.02, f"{overall_accuracy:.2f}", ha="center", fontsize=10)

  for i, (center, correct_count,
          total_count) in enumerate(zip(bin_centers, correct_counts_by_bin, total_counts_by_bin)):
    plt.text(
        center,
        accuracy_by_bin[i] + 0.02,
        f"{correct_count}/{total_count}",
        ha="center",
        fontsize=10)

  plt.xlabel("Range of Bins")
  plt.ylabel("Accuracy")
  plt.xticks(bins)
  plt.title("Accuracy by Range of Bins")
  text_str = f"ECE: {ece:.4f}\n" f"MCE: {mce:.4f}\n" f"RMSCE: {rmsce:.4f} ACE: {ace:.4f}"
  plt.text(
      0.75,
      0.6,
      text_str,
      fontsize=15,
      bbox=dict(facecolor="white", alpha=1),
      horizontalalignment="right",
      verticalalignment="top")
  plt.grid(True)
  plt.savefig(save_filename)

  response = {
      "ECE": ece,
      "MCE": mce,
      "RMSCE": rmsce,
      "ACE": ace,
  }
  return response


if __name__ == "__main__":
  args = parse_args()
  ######################################################
  project_root = Path("YOUR_PROJECT_ROOT_DIR")
  base_model_name = args.base_model_name.split("/")[-1]
  inferenced_input_path = project_root / "YOUR_INFERENCED_DIR" / f"{args.task}.{base_model_name}.inferenced.jsonl"
  save_dir = project_root / "YOUR_EVALUATE_DIR"
  save_dir.mkdir(exist_ok=True, parents=True)
  save_filename = save_dir / f"{args.task}.{base_model_name}.evaluated.jsonl"
  save_line_graph_filename = save_dir / f"{args.task}.{base_model_name}.calibration_curve.png"
  save_bar_graph_filename = save_dir / f"{args.task}.{base_model_name}.accuracy_by_bins.png"
  ######################################################

  print(f"Input file: {inferenced_input_path}")
  line_graph_data = evaluate_line_graph(str(inferenced_input_path), save_line_graph_filename)
  bar_graph_data = evaluate_bar_graph(str(inferenced_input_path), 10, save_bar_graph_filename)
  line_graph_data[0].update(bar_graph_data)
  line_graph_data[0]["calibration_curve"] = str(line_graph_data[0]["calibration_curve"])
  save_jsonl(save_filename, line_graph_data)
  print(f"Output JSON file: {save_dir / save_filename}")

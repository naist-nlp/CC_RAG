import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
import argparse
import json

####################################
project_root = Path("ROOT_PATH")
####################################

def load_jsonl(file_path):
  with open(file_path, 'r') as f:
    return [json.loads(line) for line in f]

def parse_args():
  parser = argparse.ArgumentParser(description="")
  parser.add_argument('--base_model', type=str)
  return parser.parse_args()


def plot_as_scatter(base_model):
  ########################################################
  evaluated_dir = project_root / "YOUR_EVALUATED_DIR"
  manipulated_evaluated_dir = project_root / "YOUR_MANIPULATED_EVALUATED_DIR"
  base_model_suffix = base_model.split("/")[-1]
  base_line_file = evaluated_dir / f"medmcqa.{base_model_suffix}.evaluated.jsonl"
  ########################################################

  data = []
  labels = ["Base Line"]

  try:
    base_line_data = load_jsonl(base_line_file)
  except:
    base_line_data = [{"ECE": -1, "MCE": -1, "RMSCE": -1, "accuracy": -1}]

  data.append({
      "Model Variation": "Base Line",
      "Prompt Pattern": "Base Line",
      "ECE": base_line_data[0]["ECE"],
      "MCE": base_line_data[0]["MCE"],
      "RMSCE": base_line_data[0]["RMSCE"],
      "Accuracy (%)": base_line_data[0]["accuracy"] * 100
  })

  prompt_patterns = ["1", "2", "3"]
  for prompt_pattern in prompt_patterns:
    ans1_filename = manipulated_evaluated_dir / f"ans1.prompt{prompt_pattern}.{base_model_suffix}.evaluated.jsonl"
    ans1_other2_filename = manipulated_evaluated_dir / f"ans1.other2.prompt{prompt_pattern}.{base_model_suffix}.evaluated.jsonl"
    other3_filename = manipulated_evaluated_dir / f"other3.prompt{prompt_pattern}.{base_model_suffix}.evaluated.jsonl"
    try:
      ans1_data = load_jsonl(ans1_filename)
    except:
      ans1_data = [{"ECE": -1, "MCE": -1, "RMSCE": -1, "accuracy": -1}]
    try:
      ans1_other2_data = load_jsonl(ans1_other2_filename)
    except:
      ans1_other2_data = [{"ECE": -1, "MCE": -1, "RMSCE": -1, "accuracy": -1}]
    try:
      other3_data = load_jsonl(other3_filename)
    except:
      other3_data = [{"ECE": -1, "MCE": -1, "RMSCE": -1, "accuracy": -1}]

    if ans1_data[0]["accuracy"] != -1:
      data.append({
          "Model Variation": f"ans1.prompt{prompt_pattern}",
          "Prompt Pattern": "ans1",
          "ECE": ans1_data[0]["ECE"],
          "MCE": ans1_data[0]["MCE"],
          "RMSCE": ans1_data[0]["RMSCE"],
          "Accuracy (%)": ans1_data[0]["accuracy"] * 100
      })
    if ans1_other2_data[0]["accuracy"] != -1:
      data.append({
          "Model Variation": f"ans1.other2.prompt{prompt_pattern}",
          "Prompt Pattern": "ans1.other2",
          "ECE": ans1_other2_data[0]["ECE"],
          "MCE": ans1_other2_data[0]["MCE"],
          "RMSCE": ans1_other2_data[0]["RMSCE"],
          "Accuracy (%)": ans1_other2_data[0]["accuracy"] * 100
      })
    if other3_data[0]["accuracy"] != -1:
      data.append({
          "Model Variation": f"other3.prompt{prompt_pattern}",
          "Prompt Pattern": "other3",
          "ECE": other3_data[0]["ECE"],
          "MCE": other3_data[0]["MCE"],
          "RMSCE": other3_data[0]["RMSCE"],
          "Accuracy (%)": other3_data[0]["accuracy"] * 100
      })
  df = pd.DataFrame(data)

  unique_models = df["Prompt Pattern"].unique()
  palette = sns.color_palette("colorblind", len(unique_models))
  color_dict = dict(zip(unique_models, palette))
  save_dir = project_root / 'scatter.manipulated'
  save_dir.mkdir(exist_ok=True, parents=True)

  plt.figure(figsize=(16, 9))
  for prompt_pattern in prompt_patterns:
    ans1_subset_df = df[df["Model Variation"] == f"ans1.prompt{prompt_pattern}"]
    if ans1_subset_df["Accuracy (%)"].max() == -1:
      print(f"Skipping plot for ans1.prompt{prompt_pattern} due to missing data")
      continue

    sns.scatterplot(
        data=ans1_subset_df,
        x="Accuracy (%)",
        y="ECE",
        hue="Prompt Pattern",
        palette=color_dict,
        s=200,
        marker="o")

    ans1_other2_subset_df = df[df["Model Variation"] == f"ans1.other2.prompt{prompt_pattern}"]
    if ans1_other2_subset_df["Accuracy (%)"].max() == -1:
      print(f"Skipping plot for ans1.other2.prompt{prompt_pattern} due to missing data")
      continue

    sns.scatterplot(
        data=ans1_other2_subset_df,
        x="Accuracy (%)",
        y="ECE",
        hue="Prompt Pattern",
        palette=color_dict,
        s=200,
        marker="x")

    other3_subset_df = df[df["Model Variation"] == f"other3.prompt{prompt_pattern}"]
    if other3_subset_df["Accuracy (%)"].max() == -1:
      print(f"Skipping plot for other3.prompt{prompt_pattern} due to missing data")
      continue

    sns.scatterplot(
        data=other3_subset_df,
        x="Accuracy (%)",
        y="ECE",
        hue="Prompt Pattern",
        palette=color_dict,
        s=200,
        marker="^")

    for i in range(len(ans1_subset_df)):
      x = ans1_subset_df.iloc[i]["Accuracy (%)"]
      y = ans1_subset_df.iloc[i]["ECE"]
      plt.plot([x, x], [0, y], color=color_dict["ans1"], linestyle="--", linewidth=1)

    for i in range(len(ans1_other2_subset_df)):
      x = ans1_other2_subset_df.iloc[i]["Accuracy (%)"]
      y = ans1_other2_subset_df.iloc[i]["ECE"]
      plt.plot([x, x], [0, y], color=color_dict["ans1.other2"], linestyle="--", linewidth=1)

    for i in range(len(other3_subset_df)):
      x = other3_subset_df.iloc[i]["Accuracy (%)"]
      y = other3_subset_df.iloc[i]["ECE"]
      plt.plot([x, x], [0, y], color=color_dict["other3"], linestyle="--", linewidth=1)

  plt.title(f"{base_model_suffix} - Combined Top-k and Length Variations", fontsize=20)
  plt.xlabel("Accuracy (%)", fontsize=16)
  plt.ylabel("ECE", fontsize=16)
  plt.legend(loc='upper right', title="Prompt Pattern & Configuration", bbox_to_anchor=(1.2, 1))
  plt.tight_layout()
  plt.grid(True)

  plot_file = save_dir / f"{base_model_suffix}.png"
  plt.savefig(plot_file)
  print(f"Saved combined plot: {plot_file}")

if __name__ == "__main__":
  args = parse_args()
  plot_as_scatter(args.base_model)

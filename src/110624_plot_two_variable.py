import seaborn as sns
import matplotlib.pyplot as plt
import json
from pathlib import Path
import os
import argparse

project_root = Path('/cl/home2/shintaro/rag-notebook')


def load_jsonl(file_path):
  with open(file_path, 'r') as f:
    return [json.loads(line) for line in f]


def plot_as_scatter(base_model, task):
  evaluated_dir = project_root / 'make_datastore_py310/data/evaluated'
  base_model_suffix = base_model.split("/")[-1]
  base_line_file = evaluated_dir / f"{task}.{base_model_suffix}.evaluated.jsonl"

  rag_models = ["ncbi/MedCPT-Query-Encoder", "facebook/contriever", "allenai/specter", "BM25"]
  rag_dbs = ["statpearls", "textbooks"]

  data = []

  try:
    base_line_data = load_jsonl(base_line_file)
  except:
    base_line_data = [{"ECE": -1, "accuracy": -1}]

  data.append({
      "Model Variation":
          "Base Line",
      "RAG Model":
          "Base Line",
      "ECE":
          base_line_data[0]["ECE"],
      "Accuracy (%)":
          base_line_data[0]["accuracy"] * 100 if base_line_data[0]["accuracy"] != -1 else -1
  })
  blind_pallete = sns.color_palette("colorblind", 10)

  # color blindを考慮したカラーパレット
  color_pallet_dict = {
      "MedCPT-Query-Encoder": blind_pallete[0],  # 青
      "contriever": blind_pallete[2],  # 緑
      "specter": blind_pallete[1],  # 濃いオレンジ
      "BM25":
          blind_pallete[4]  # 紫
  }

  for rag_model in rag_models:
    for rag_db in rag_dbs:
      for top_k in [1, 3]:
        for length in [256, 512]:
          rag_model_suffix = rag_model.split("/")[-1]
          model_variant = f"{rag_model_suffix} ({rag_db}, Top-{top_k}, {length} length)"

          top_file = evaluated_dir / f"{task}.{rag_db}.{rag_model_suffix}.{length}length.{base_model_suffix}.top{top_k}.evaluated.jsonl"

          try:
            top_data = load_jsonl(top_file)
          except:
            top_data = [{"ECE": -1, "accuracy": -1}]

          if top_data[0]["accuracy"] != -1:
            data.append({
                "Model Variation": model_variant,
                "RAG Model": rag_model_suffix,
                "ECE": top_data[0]["ECE"],
                "Accuracy (%)": top_data[0]["accuracy"] * 100
            })

  # `Accuracy (%)`と`ECE`の範囲を計算
  accuracy_values = [entry["Accuracy (%)"] for entry in data if entry["Accuracy (%)"] != -1]
  ece_values = [entry["ECE"] for entry in data if entry["ECE"] != -1]

  accuracy_min, accuracy_max = min(accuracy_values), max(accuracy_values)
  ece_min, ece_max = min(ece_values), max(ece_values)

  # 線の範囲を相対的に設定
  line_x = [accuracy_min, accuracy_max]
  line_y = [ece_max, ece_min]

  # 描画
  plt.figure(figsize=(16, 9))
  unique_models = list(set([entry["RAG Model"] for entry in data]))
  palette = sns.color_palette("colorblind", len(unique_models))

  for top_k in [1, 3]:
    for length in [256, 512]:
      subset = [
          entry for entry in data if f"Top-{top_k}, {length} length" in entry["Model Variation"] or
          entry["Model Variation"] == "Base Line"
      ]

      if all(entry["Accuracy (%)"] == -1 for entry in subset):
        print(f"Skipping plot for Top-{top_k}, {length} length due to missing data")
        continue

      marker = "o" if (top_k == 1 and length == 256) else \
               "x" if (top_k == 3 and length == 256) else \
               "s" if (top_k == 1 and length == 512) else "D"

      for entry in subset:
        if entry["Accuracy (%)"] != -1:
          if entry["Model Variation"] == "Base Line":
            # Base Line の特別なスタイル
            plt.scatter(
                x=entry["Accuracy (%)"],
                y=entry["ECE"],
                color="red",
                s=1000,
                marker="*",
                label="Base Line")
          else:
            # その他のデータポイント
            plt.scatter(
                x=entry["Accuracy (%)"],
                y=entry["ECE"],
                color=color_pallet_dict[entry["RAG Model"]],
                s=500,
                marker=marker,
            )

  # 相対的な線を描画
  plt.plot(line_x, line_y, color="gray", linestyle="--", linewidth=3)
  plt.tight_layout()
  plt.grid(True)

  save_dir = project_root / 'scatter'
  save_dir.mkdir(exist_ok=True, parents=True)
  plot_file = save_dir / f"{task}_{base_model_suffix}.png"
  plt.savefig(plot_file)
  print(f"Saved combined plot: {plot_file}")


def parse_args():
  parser = argparse.ArgumentParser(description="")
  parser.add_argument('--base_model', type=str)
  parser.add_argument('--task', type=str)
  return parser.parse_args()


if __name__ == "__main__":
  args = parse_args()
  plot_as_scatter(args.base_model, args.task)

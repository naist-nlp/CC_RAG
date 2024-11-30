from pathlib import Path
import os
import math
import json
import torch
from dotenv import load_dotenv
import argparse
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from prompt import *
import logging
import sys
import random
from tqdm import tqdm

logging.basicConfig(
    format="| %(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO",
    stream=sys.stdout,
)
logger: logging.Logger = logging.getLogger(__name__)

###############################################
project_root = Path("ROOT_PATH")
load_dotenv()
random.seed(42)
###############################################

def load_jsonl(file_path):
  with open(file_path, "r") as f:
    return [json.loads(line) for line in f]


def save_jsonl(file_path, data):
  with open(file_path, "w") as f:
    for line in data:
      f.write(json.dumps(line) + "\n")


def calculate_probability(args, model, tokenizer, sentence_prefix, answer):
  prefix_inputs = tokenizer(sentence_prefix, return_tensors="pt", padding=False).to(args["device"])
  prefix_inputs["input_ids"] = prefix_inputs["input_ids"][:, 0:]
  prefix_inputs["attention_mask"] = prefix_inputs["attention_mask"][:, 0:]
  prefix_length = prefix_inputs["input_ids"].size(1)

  answer_inputs = tokenizer(answer, return_tensors="pt", padding=False).to(args["device"])
  answer_inputs["input_ids"] = answer_inputs["input_ids"][:, 0:]
  answer_inputs["attention_mask"] = answer_inputs["attention_mask"][:, 0:]
  answer_length = answer_inputs["input_ids"].size(1)

  sentence_inputs = torch.cat([prefix_inputs["input_ids"], answer_inputs["input_ids"]], dim=1)
  sentence_attention_mask = torch.cat(
      [prefix_inputs["attention_mask"], answer_inputs["attention_mask"]], dim=1)

  with torch.no_grad():
    outputs = model(input_ids=sentence_inputs, attention_mask=sentence_attention_mask)

  logits = outputs.logits
  shift_logits = logits[..., :-1, :].contiguous()
  shift_labels = sentence_inputs[..., 1:].contiguous()
  sentence_attention_mask[..., :prefix_length] = 0

  loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_token_id)
  loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
  loss = (loss.view(shift_labels.size()) * sentence_attention_mask[..., 1:].contiguous())
  loss = loss.sum() / sentence_attention_mask[..., 1:].sum()

  average_loss = loss.mean()
  log_prob = -average_loss.item()
  assert math.isnan(log_prob) == False, f"log_prob is nan: {log_prob}"
  return log_prob


def get_random_sentence(data, k, answer):
  retrieved_documents = []
  for _ in range(k):
    while True:
      line = random.choice(data)
      if line['explanation'] is not None and not answer in line['explanation']:
        retrieved_documents.append(line['explanation'])
        break
  return retrieved_documents


def make_prompt(question, choices, retrieved_documents, prompt_pattern):
  system_prompt = ""
  concatenated_text = "* " + "\n* ".join(retrieved_documents)
  system_prompt = general_medrag_system
  if prompt_pattern == 1:
    template = general_medrag_pattern1
  elif prompt_pattern == 2:
    template = general_medrag_pattern2
  else:
    template = general_medrag_pattern3
  prompt = template.format(
      context=concatenated_text,
      question=question,
      option_1=choices[0],
      option_2=choices[1],
      option_3=choices[2],
      option_4=choices[3],
  )
  prompt = system_prompt + prompt
  return prompt


def initialize_model(model_name, quantize_type, device, hf_token):
  tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True)

  if tokenizer.pad_token is None:
    if tokenizer.eos_token is None:
      tokenizer.add_special_tokens({"pad_token": "<pad>"})
    else:
      tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

  model_kwargs = {
      "low_cpu_mem_usage": True,
      "trust_remote_code": True,
      "token": hf_token,
  }

  if quantize_type == "4bit":
    model_kwargs.update({
        "quantization_config":
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
        "torch_dtype":
            torch.float16,
    })
  elif quantize_type == "8bit":
    model_kwargs.update({
        "device_map": "auto",
        "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
        "use_cache": True,
    })
  elif quantize_type == "half":
    model_kwargs.update({"torch_dtype": torch.float16})

  model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
  # 量子化がない場合、モデルを適切なデバイスに移動
  if quantize_type == "none" or quantize_type == "half":
    model.to(device)
  model.eval()
  return model, tokenizer


def run_model(base_model_name, qa_dataset_path, output_file, quantize_type, prompt_pattern):
  if base_model_name == "meta-llama/Llama-2-70b" or base_model_name == "meta-llama/Llama-3.1-8B":
    hf_token = os.environ["GEMMA_TOKEN"]
  else:
    hf_token = os.environ["HUGGINGFACE_TOKEN"]

  args = {
      "model_path": base_model_name,
      "hf_token": hf_token,
      "quantize_type": quantize_type,
      "device": "cuda" if torch.cuda.is_available() else "cpu",
      "max_length": 2048,
      "seed": 42,
  }
  model, tokenizer = initialize_model(args["model_path"], args["quantize_type"], args["device"],
                                      args["hf_token"])

  result_dataset = []
  data = load_jsonl(qa_dataset_path)
  # explanation がnullのものは除外する。2206件になるはず
  data = [line for line in data if line['explanation'] is not None]

  for item in tqdm(data):
    question = item["question"]
    choices = item["choices"]["text"]
    labels = item["choices"]["label"]

    log_probs = {}

    choise_perplexity_probabilities = []
    random_noise = get_random_sentence(data, 3, choices[labels.index(item["answerKey"])])
    for i, choice in enumerate(choices):
      retrieved_documents = random_noise
      sentence_prefix = make_prompt(question, choices, retrieved_documents, prompt_pattern)
      answer = labels[i]

      log_prob = calculate_probability(args, model, tokenizer, sentence_prefix, answer)
      log_probs[choice] = log_prob

    log_probs_list = torch.Tensor(list(log_probs.values()))
    lse = torch.logsumexp(log_probs_list, 0).item()
    probabilities = {k: math.exp(v - lse) for k, v in log_probs.items()}

    for choice, probability in probabilities.items():
      choise_perplexity_probabilities.append({
          "choice": choice,
          "probability": probability,
      })
    best_label = max(probabilities, key=probabilities.get)
    best_choise_idx = choices.index(best_label)
    best_choise_label = labels[best_choise_idx]
    is_correct = item["answerKey"] == best_choise_label

    response = {
        "id": item["id"],
        "question": item["question"],
        "choices": {
            "text": choices,
            "label": labels
        },
        "answerKey": item["answerKey"],
        "probabilities": probabilities,
        "best_choice": {
            "best_label": best_choise_label,
            "best_probability": probabilities[best_label],
        },
        "is_correct": is_correct,
    }
    result_dataset.append(response)
  save_jsonl(output_file, result_dataset)
  print(f"Results saved to {output_file}!")


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--inference_model", type=str, required=True, default="microsoft/Phi-3.5-mini-instruct")
  parser.add_argument("--quantize_type", type=str, default="none")
  parser.add_argument("--prompt_pattern", type=int)
  return parser.parse_args()


if __name__ == "__main__":
  load_dotenv()
  args = parse_args()
 #####################################################
  model_input_file = project_root / "YOUR_RETRIEVED_DIR" / "medmcqa.pubmed.BM25.512length.retrieved.jsonl"
  model_output_dir = project_root / "YOUR_OUTPUT_DIR"
  model_output_dir.mkdir(exist_ok=True, parents=True)
  model_output_file = model_output_dir / f"other3.prompt{args.prompt_pattern}.{args.inference_model.split('/')[-1]}.manipulated.jsonl"
  ####################################################

  logger.info(f"Running inference using {args.inference_model} model...")
  logger.info(f"Retrieved documents are loaded from {model_input_file}")
  logger.info(f"Results will be saved to {model_output_file}")
  run_model(args.inference_model, model_input_file, model_output_file, args.quantize_type,
            args.prompt_pattern)

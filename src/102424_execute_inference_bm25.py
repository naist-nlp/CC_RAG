from pathlib import Path
import os
import math
import json
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import Stemmer
import bm25s
import torch
import logging
import argparse
import sys
from prompt import *
from dotenv import load_dotenv

logging.basicConfig(
    format="| %(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO",
    stream=sys.stdout,
)
logger: logging.Logger = logging.getLogger(__name__)

def load_jsonl(file_path):
  with open(file_path, "r") as f:
    return [json.loads(line) for line in f]


def save_jsonl(file_path, data):
  with open(file_path, "w") as f:
    for line in data:
      f.write(json.dumps(line) + "\n")


def search_by_bm25(query_text, k, retriever, corpus):
  stemmer = Stemmer.Stemmer("english")
  query_tokens = bm25s.tokenize([query_text], stopwords="en", stemmer=stemmer)
  results, scores = retriever.retrieve(query_tokens, k=k)
  results = [str(doc_id) for doc_id in results[0]]
  return results


def setup_bm25_retriever(index_path):
  retriever = bm25s.BM25.load(index_path, load_corpus=True)
  corpus_dict = {str(doc["id"]): doc["text"] for doc in retriever.corpus}
  return retriever, corpus_dict


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
  if quantize_type == "none" or quantize_type == "half":
    model.to(device)
  model.eval()
  return model, tokenizer


def make_prompt(task, question, choices, retrieved_documents):
  concatenated_text = "* " + "\n* ".join(retrieved_documents)
  system_prompt = ""
  if task in ["medqa", "medmcqa", "mmlu", "pubmedqa"]:
    system_prompt = general_medrag_system
    if task == "pubmedqa":
      prompt = pubmedqa_medrag.format(
          context=concatenated_text,
          question=question,
          option_1=choices[0],
          option_2=choices[1],
          option_3=choices[2],
      )
    else:
      prompt = general_medrag.format(
          context=concatenated_text,
          question=question,
          option_1=choices[0],
          option_2=choices[1],
          option_3=choices[2],
          option_4=choices[3],
      )
    prompt = system_prompt + prompt
  else:
    prompt = f"""Here is the question:\n{question}\n\nHere are the potential choices:\nA: {choices[0]}\nB: {choices[1]}\nC: {choices[2]}\nD: {choices[3]}\n\nPlease think step-by-step and generate your output in below format:\nAnswer: """
  return prompt


def run_model(base_model_name, qa_dataset_path, output_file, task, top_k, inference_max_length,
              rag_db, max_length, quantize_type):
  if base_model_name == "meta-llama/Llama-2-70b" or base_model_name == "meta-llama/Llama-3.1-70B-Instruct":
    hf_token = os.environ["GEMMA_TOKEN"]
  else:
    hf_token = os.environ["HUGGINGFACE_TOKEN"]

  args = {
      "model_path": base_model_name,
      "hf_token": hf_token,
      "quantize_type": quantize_type,
      "device": "cuda" if torch.cuda.is_available() else "cpu",
      "max_length": inference_max_length,
      "seed": 42,
  }

  model, tokenizer = initialize_model(args["model_path"], args["quantize_type"], args["device"],
                                      args["hf_token"])

  result_dataset = []
  data = load_jsonl(qa_dataset_path)

  ###########################################################
  # Please set the following variables
  datastore_dir = project_root /  "YOUR_INDEX_DIR" / f"{rag_db}.bm25.{max_length}length.datastore"
  ###########################################################
  retriever, corpus = setup_bm25_retriever(datastore_dir)

  for item in tqdm(data):
    question = item["question"]
    choices = item["choices"]["text"]
    labels = item["choices"]["label"]
    log_probs = {}

    choise_perplexity_probabilities = []
    for i, choice in enumerate(choices):
      retrived_documents = search_by_bm25(question, top_k, retriever, corpus)
      sentence_prefix = make_prompt(task, question, choices, retrived_documents)
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
  parser.add_argument("--inference_model", type=str, required=True, help="Model name to run inference")
  parser.add_argument("--top_k", type=int, default=5, help="Top-k retrieved documents")
  parser.add_argument("--rag_db", type=str, default="textbooks", help="RAG dataset name")
  parser.add_argument("--task", type=str, required=True, default="medqa", help="Task name")
  parser.add_argument("--rag_max_length", type=int, required=True, default=512, help="RAG max embedding length")
  parser.add_argument("--inference_max_length", type=int, required=True, default=2048, help="Inference max length")
  parser.add_argument("--quantize_type", type=str, default="none")
  return parser.parse_args()


if __name__ == "__main__":
  args = parse_args()
  load_dotenv()
  ###############################################
  # Please set the following variables
  project_root = Path("YOUR_PROJECT_ROOT_DIR")
  model_input_file = project_root / "YOUR_MODEL_INPUT_DIR" / f"{args.task}.{args.rag_db}.BM25.{args.rag_max_length}length.retrieved.jsonl"
  model_output_dir = project_root / "YOU_MODEL_OUTPUT_DIR"
  model_output_dir.mkdir(exist_ok=True, parents=True)
  model_output_file = model_output_dir / f"{args.task}.{args.rag_db}.BM25.{args.rag_max_length}length.{args.inference_model.split('/')[-1]}.top{args.top_k}.inferenced.jsonl"
  ###############################################

  logger.info(
      f"Running inference for {args.task} task with {args.rag_db} dataset using {args.inference_model} model..."
  )
  logger.info(f"Retrieved documents are loaded from {model_input_file}")
  logger.info(f"Results will be saved to {model_output_file}")
  run_model(args.inference_model, model_input_file, model_output_file, args.task, args.top_k,
            args.inference_max_length, args.rag_db, args.rag_max_length, args.quantize_type)

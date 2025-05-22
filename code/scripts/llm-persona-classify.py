import os
import sys
import pickle
import argparse
import json_repair

# Load env from ../.env
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"    

PROMPT_CHOICES = [
    "internal_monologue",
    "perspective_taking",
    "cognitive_empathy",
    "decision_making",
    "contextual_persona",
    "moral_dilemma",
    "immersive_first_person",
    "think_aloud",
    "naive"
]

parser = argparse.ArgumentParser(description="Set CUDA_VISIBLE_DEVICES")
parser.add_argument("--cuda_devices", type=str, default="1", help="CUDA devices to be used")
parser.add_argument("--model_id", type=str, required=True, help="Model ID to be used")
parser.add_argument("--prompt", type=str, choices=PROMPT_CHOICES, required=True, help="Type of prompt to use for persona simulation")
parser.add_argument("--task", type=str, required=True, help="Task to do (S6, S100, Prolific, or a combination of them)")

args = parser.parse_args()
to_be_executed = args.task.split(",")

# Check if the output file already exists
# for dataset in to_be_executed:
#     for filename in os.listdir("../data/output/"):
#         if filename == f"{dataset}_{args.model_id.split('/')[-1]}_{args.prompt}.pkl":
#             print(f"Output file {filename} already exists. Exiting.")
#             exit(0)

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

import json
import logging
import pandas as pd

from tqdm import tqdm
from typing import Literal
from pydantic import BaseModel
from dotenv import load_dotenv
from collections import defaultdict


if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

load_dotenv(dotenv_path="../.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Log to console
    ]
)

tqdm.pandas()

class LLMClassification_Scheme(BaseModel):
    label: Literal[0,1]
    explanation: str


# Reads and loads the personas.json file from the specified data path.
def load_personas(data_path):
    personas_path = os.path.join(data_path, "processed", "personas.json")
    try:
        with open(personas_path, "r", encoding="utf-8") as f:
            personas = json.load(f)
        print(f"Loaded personas from {personas_path}")
        return personas
    except FileNotFoundError:
        print(f"File not found at {personas_path}")
        return None
    

def build_llm(model_id, cache_path):
    print(f"Loading {model_id}")
    try:
        return LLM(
            model                   = model_id,
            tokenizer_mode          = "mistral" if "mistral" in model_id else "auto",
            task                    = "generate",
            trust_remote_code       = True,
            enable_prefix_caching   = True,
            max_model_len           = 9216,
            download_dir            = cache_path,
            gpu_memory_utilization  = 0.95,
            guided_decoding_backend = "xgrammar:disable-any-whitespace",
            tensor_parallel_size    = os.environ["CUDA_VISIBLE_DEVICES"].count(",") + 1
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

def save_inference_results(persona_id, row, relevance_obj, excluded_columns=["speaker", "text"]):
    # Extract row data excluding specific columns
    try:
        filtered_row = row.drop(labels=excluded_columns).to_dict()
        # Add persona ID
        filtered_row["worker_id"] = persona_id
        # Add LLM response
        filtered_row.update({
            "model_answer" : relevance_obj.label,
            "explanation"  : relevance_obj.explanation
        })
        return filtered_row
    except Exception as e:
        print(f"Failed to save inference results: {e}")
        return {}
    
def generate_prompt_variables(persona, query):
    keys = [
        "consideration", "political_views", "age",
        "environment", "school", "taxes", "southern_border"
    ]
    return {key: persona.get(key, "unspecified") for key in keys} | {"query": query}
    

def main():
    model_id = args.model_id
    # Retrieve variables from the .env file
    cache_path   = os.environ.get("cache_path")
    ground_truth = os.environ.get("ground_truth")

    with open(f"../{ground_truth}", "rb") as f:
        ground_truth_df = pickle.load(f)

    personas = load_personas("../data/")
    llm = build_llm(model_id, cache_path)
    
    with open("prompts.json", "r") as f:
        prompts = json.load(f)
        template = prompts.get(args.prompt)

    sampling = SamplingParams(
        temperature=0.1,
        max_tokens=700,
        guided_decoding=GuidedDecodingParams(LLMClassification_Scheme.model_json_schema()),
    )

    print("Data Preprocessing Step Done.")

    results = []
    all_prompts = []
    metadata = []
    # Loop over each task and then over each persona and statement to build the batch.
    for this_dataset_execution in to_be_executed:
        # Filter data for the current task
        filtered_personas = [p for p in personas if p.get("task") == this_dataset_execution]
        filtered_ground_truth_df = ground_truth_df[ground_truth_df["task"] == this_dataset_execution]
    
        for this_persona in tqdm(filtered_personas, desc=f"Task: {this_dataset_execution} Personas"):
            # Get ground truth rows for the current persona
            this_persona_df = filtered_ground_truth_df[filtered_ground_truth_df["worker_id"] == this_persona["id"]]
    
            for _, row in this_persona_df.iterrows():
                # Format variables for the current prompt
                prompt_variables = {
                    "consideration": this_persona.get("consideration", "unspecified"),
                    "political_views": this_persona.get("political_views", "unspecified"),
                    "age": this_persona.get("age", "unknown"),
                    "environment": this_persona.get("environment", "neutral"),
                    "school": this_persona.get("school", "unknown"),
                    "taxes": this_persona.get("taxes", "unknown"),
                    "southern_border": this_persona.get("southern_border", "no opinion"),
                    "query": row["text"],
                }
    
                # Create the filled prompt and save its metadata
                filled_prompt = template.format(**prompt_variables)
                all_prompts.append(filled_prompt)
                metadata.append({
                    "task": this_dataset_execution,
                    "persona_id": this_persona["id"],
                    "row": row  # row already contains the statement and other info
                })

    
    # Generate responses in one batch
    generations = llm.generate(all_prompts, sampling, use_tqdm=True)
    
    # Process results and group them by the task
    results_by_task = defaultdict(list)
    
    for meta, generation in zip(metadata, generations):
        raw_text = generation.outputs[0].text
        
        # Attempt to parse the JSON output
        try:
            relevance_obj = LLMClassification_Scheme(**json.loads(raw_text))
        except json.decoder.JSONDecodeError:
            # Fallback: try to patch common issues by appending an ending if needed
            try:
                fixed_json = json_repair.loads(raw_text)
                if isinstance(fixed_json, list):
                    fixed_json = fixed_json[0]

                relevance_obj = LLMClassification_Scheme(**fixed_json)
            except json.decoder.JSONDecodeError:
                print(f"Failed to parse JSON: {raw_text}")
                continue
            except Exception as e:
                print(f"Error during JSON repair: {e}")
                print(f"Original text: {raw_text}")
                print(json_repair.loads(raw_text))
                exit()
    
        # Use the saved metadata to create and store the result
        result = save_inference_results(meta["persona_id"], meta["row"], relevance_obj)
        results_by_task[meta["task"]].append(result)
    
    # Save results per task
    for task, results in results_by_task.items():
        results_df = pd.DataFrame(results)
        output_filename = f"../data/output/{task}_{model_id.split('/')[-1]}_{args.prompt}.pkl"
        results_df.to_pickle(output_filename)
    
    print("Done.")
    sys.exit(0)


if __name__ == "__main__":
    main()
    sys.exit(0)
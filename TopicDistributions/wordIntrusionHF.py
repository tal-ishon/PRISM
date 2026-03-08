from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import random
import json
import os
import sys
import torch
import re
from tqdm import tqdm

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not set. Export it before running.")
# os.environ["HUGGINGFACE_API_TOKEN"] = "TI_access_model"

# Set custom cache directory for Hugging Face resources
os.environ["TRANSFORMERS_CACHE"] = "/data/users/ishonta/cache/models"
os.environ["HF_HOME"] = "/data/users/ishonta/cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "3" # Choose GPU to run on

# # Initialize the LLM pipeline
# model_name = "meta-llama/Llama-3.3-70B-Instruct"

# # Load the model with specified settings
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16,   # Use bfloat16 for reduced memory usage
#     device_map="auto",           # Automatically map model across GPUs
#     cache_dir=os.environ["TRANSFORMERS_CACHE"],  # Explicit cache directory
# )

# # Load the tokenizer
# tokenizer = AutoTokenizer.from_pretrained(
#     model_name,
#     cache_dir=os.environ["TRANSFORMERS_CACHE"],  # Explicit cache directory
# )

# # Ensure the tokenizer has a padding token
# if tokenizer.pad_token_id is None:
#     tokenizer.pad_token_id = tokenizer.eos_token_id

# Define generation settings
temp = 0.7

# # Initialize the text-generation pipeline
# llm_model = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     device_map="auto",            # Automatically map model across GPUs
#     model_kwargs={"torch_dtype": torch.bfloat16},  # Use bfloat16
# )

def initialize_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=os.environ["TRANSFORMERS_CACHE"]
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=os.environ["TRANSFORMERS_CACHE"]
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        model_kwargs={"torch_dtype": torch.bfloat16}
    )
    return llm_pipeline

model_name = "meta-llama/Llama-3.3-70B-Instruct"
llm_model = initialize_model(model_name)

# -------------------------
# Load Data from CSV Files
# -------------------------

def load_csv(filepath):
    return pd.read_csv(filepath)

# -------------------------
# Prepare Data for Tasks
# -------------------------

def get_top_words_for_topics(topic_word_df, top_n=8):
    top_words_by_topic = {}
    for topic_id in topic_word_df['Topic']:
        # Get the row corresponding to the current topic (excluding the 'Topic' column)
        topic_row = topic_word_df.loc[topic_word_df['Topic'] == topic_id].drop(columns='Topic').iloc[0]
        # Sort the row values in descending order and get the top N columns (words)
        top_words = topic_row.sort_values(ascending=False).head(top_n)
        
        # Collect the top N words (column names)
        top_words_by_topic[topic_id] = top_words.index.tolist()
    
    return top_words_by_topic

def get_bottom_words_for_topics(topic_word_df, top_n=10):
    top_words_by_topic = {}
    for topic_id in topic_word_df['Topic']:
        # Get the row corresponding to the current topic (excluding the 'Topic' column)
        topic_row = topic_word_df.loc[topic_word_df['Topic'] == topic_id].drop(columns='Topic').iloc[0]
        # Sort the row values in descending order and get the top N columns (words)
        top_words = topic_row.sort_values(ascending=True).head(top_n)
        
        # Collect the top N words (column names)
        top_words_by_topic[topic_id] = top_words.index.tolist()
    
    return top_words_by_topic


def get_doc_topics(doc_topic_df, prob_threshold=0.0):
    doc_topics = {}
     # Iterate through each document (row)
    for doc_id, row in doc_topic_df.iterrows():
        # Exclude the 'Document' column (first column) and get the topic probabilities
        topic_probabilities = row.drop('Document')
        # Identify topics where the probability exceeds the threshold
        topics_above_threshold = topic_probabilities[topic_probabilities > prob_threshold].index.tolist()
        # Convert the topic names (strings) to integers if necessary (as column names are strings)
        topics_above_threshold = [int(topic) for topic in topics_above_threshold]
        # Store the result in the dictionary
        doc_topics[doc_id] = topics_above_threshold
    
    return doc_topics

# -------------------------
# Word Intrusion Task
# -------------------------

def word_intrusion(top_words_by_topic, model, save_for_human_eval=False):
    word_intrusion_results = []

    for topic_id, words in top_words_by_topic.items():
        
        # Add an intruder word that does not belong
        all_other_words = sum(bottom_words_by_topic.values(), [])
        intruder_word = random.choice([word for word in all_other_words if word not in words])
        
        # Setup words including intruder
        word_list = words + [intruder_word]
        random.shuffle(word_list)
        
        numbered_word_list = ""
        for j, word in enumerate(word_list):
                numbered_word_list += f"{j + 1}. {word}\r\n"

        input_text = f"From the following list of tokens, identify the one token that does not belong with the others. For example: for\r\n1. Banana\r\n2. Orange\r\n3. Japan\r\n4. Strawberry\r\n5. Tree\r\nI expect the answer 3. Here are the words:\r\n {numbered_word_list}.\n In your response, provide only the intruder word's index without any additional explanation"

        messages = [
            {"role": "user", "content": f"{input_text}"},
        ]
        # Initialize pipeline with the loaded model

        # Generate output with constraints
        result = pipeline(
            messages,
            max_new_tokens=20,  # Limit to a few tokens to get a short response
            no_repeat_ngram_size=2,
            return_full_text=False,  # Only show generated text, not the prompt
            temperature = temp
        )

        clean_result = result[0]['generated_text'].strip()
        answer_ix = int(re.sub(r'[^0-9]', '', clean_result)) - 1
        word_result = word_list[answer_ix]
        intrusion_result = {
            "topic_id": topic_id,
            "prompt": input_text,
            "model_response": word_result,
            "intruder": intruder_word
        }
        
        print("Words List: {}\nModel Intuder: {}\nReal Intruder: {}".format(word_list, word_result, intruder_word))
        
    # Optionally save for human evaluation   
    if save_for_human_eval:
        with open("word_intrusion_results.json", "w") as f:
            json.dump(word_intrusion_results, f)

    return word_intrusion_results

def updated_word_intrusion(topic_word_df, intruders, model, TM_model, save_for_human_eval=False):
    word_intrusion_results = []
    topic_word_df = topic_word_df.drop(topic_word_df.columns[0], axis=1)

    for topic_id, intruder in enumerate(tqdm(intruders, desc="Processing topics")):
        words = topic_word_df.iloc[:, topic_id]
        words_list = list(words)

        numbered_word_list = ""
        for j, word in enumerate(words):
                numbered_word_list += f"{j + 1}. {word}\r\n"

        input_text = f"""From the following list of tokens, identify the one token that does not belong with the others. 
        For example: for\r\n1. Banana\r\n2. Orange\r\n3. Japan\r\n4. Strawberry\r\n5. Tree\r\nThe expected answer is index 3. 
        The reason 3 is the intruder's index is since Banana, Orange, Strawberry and Tree are related to fruits in a way, but Japan is a country.
        Another example: for\r\n1. Dog\r\n2. Cat\r\n3. Horse\r\n4. Apple\r\n5. Pig\r\nThe expected answer is index 4.
        The reason 4 is the intruder's index is since Dog, Cat, Horse and Pig are different kinds of animals, while Apple is a fruit.

        Here are your words:\r\n {numbered_word_list}.\n 
        In your response, you have to provide only one index between 1 to {len(words_list)}, where the index is the intruder word's index from the given list. Without any additional explanations."""

        messages = [
            {"role": "user", "content": f"{input_text}"},
        ]
        # Initialize pipeline with the loaded model

        # Generate output with constraints
        result = llm_model(
            messages,
            max_new_tokens=20,  # Limit to a few tokens to get a short response
            no_repeat_ngram_size=2,
            return_full_text=False,  # Only show generated text, not the prompt
            temperature = temp
        )

        try:
            clean_result = result[0]['generated_text'].strip()
            answer_ix = int(re.sub(r'[^0-9]', '', clean_result)) - 1
            word_result = words[answer_ix]
        except:
            print(clean_result)
            word_result = "ERROR"
    
        intrusion_result = {
            "Topic_id": topic_id,
            "Word List": words_list,
            "Model Response": word_result,
            "Real Intruder": intruder
        }

        word_intrusion_results.append(intrusion_result)
        
        # print("Words List: {}\nModel Intuder: {}\nReal Intruder: {}".format(words_list, word_result, intruder))

    # Optionally save for human evaluation  
    path_save = f"/home/dsi/ishonta/TopicDistributions/Evaluations/{run_number}/{dataset}/{num_topics}"
    if not os.path.exists(path_save):
        os.makedirs(path_save)        
    if save_for_human_eval:
        with open(f"{path_save}/{TM_model}_10_word_intrusion_results.json", "w") as f:
            json.dump(word_intrusion_results, f, indent=4)

    return word_intrusion_results

# -------------------------
# Topic Intrusion Task
# -------------------------

def topic_intrusion(doc_topics, top_words_by_topic, model, save_for_human_eval=False):
    topic_intrusion_results = []

    for doc_id, topics in doc_topics.items():
        all_topic_ids = list(top_words_by_topic.keys())
        intruder_topic_id = random.choice([t for t in all_topic_ids if t not in topics])
        intruder_topic_words = top_words_by_topic[intruder_topic_id]
        
        topic_words_list = [top_words_by_topic[topic_id] for topic_id in topics] + [intruder_topic_words]
        random.shuffle(topic_words_list)
        
        input_text = f"From the following list of topics, identify the one topic that does not belong with the others: {topic_words_list}. In your response, use only the intruder topic from the list without any additional explanation."
        messages = [
            {"role": "user", "content": f"{input_text}"},
        ]
        # Initialize pipeline with the loaded model

        # Generate output with constraints
        result = pipeline(
            messages,
            max_new_tokens=20,  # Limit to a few tokens to get a short response
            no_repeat_ngram_size=2,
            return_full_text=False,  # Only show generated text, not the prompt
            temperature = temp
        )

        topic_result = result[0]['generated_text'].strip()
        intrusion_result = {
            "doc_id": doc_id,
            "prompt": input_text,
            "model_response": topic_result,
            "intruder_topic_id": intruder_topic_id
        }
        
        topic_intrusion_results.append(intrusion_result)
        

    with open("topic_intrusion_results.json", "w") as f:
        json.dump(topic_intrusion_results, f)

    return topic_intrusion_results


def updated_topic_intrusion(doc_topics, top_words_by_topic, model, save_for_human_eval=False):
    topic_intrusion_results = []

    for doc_id, topics in tqdm(doc_topics.items(), desc="Processing documents"):
        all_topic_ids = list(top_words_by_topic.keys())
        intruder_topic_id = random.choice([t for t in all_topic_ids if t not in topics])
        intruder_topic_words = top_words_by_topic[intruder_topic_id]

        topic_words_list = [
            (topic_id, top_words_by_topic[topic_id]) for topic_id in topics
        ] + [(intruder_topic_id, intruder_topic_words)]
        random.shuffle(topic_words_list)

        numbered_topic_list = ""
        for i, (topic_id, topic_words) in enumerate(topic_words_list):
            numbered_topic_list += f"{i + 1}. Topic {topic_id}: {', '.join(topic_words)}\r\n"

        input_text = f"""From the following list of topics, identify the one topic that does not belong with the others.\
        For example: for\r\n1. Topic 1: Banana, Orange, Strawberry\r\n2. Topic 2: Dog, Cat, Horse\r\n3. Topic 3: Japan, China, Korea\r\n4. Topic 4: Apple, Mango, Peach\r\nThe expected answer is index 3. The reason 3 is the intruder's index is since Topic 3 contains countries, while the others contain fruits or animals.\r\n\
\r\nHere are your topics:\r\n {numbered_topic_list}.\n\
In your response, provide only one index between 1 to {len(topic_words_list)}, where the index is the intruder topic's index from the given list. Without any additional explanation"""

        messages = [
            {"role": "user", "content": f"{input_text}"},
        ]

        # Generate output with constraints
        result = model(
            messages,
            max_new_tokens=20,  # Limit to a few tokens to get a short response
            no_repeat_ngram_size=2,
            return_full_text=False,  # Only show generated text, not the prompt
            temperature=temp
        )

        clean_result = result[0]['generated_text'].strip()
        answer_ix = int(re.sub(r'[^0-9]', '', clean_result)) - 1
        model_response = topic_words_list[answer_ix]

        intrusion_result = {
            "doc_id": doc_id,
            "Topics List": [{"Topic ID": tid, "Words": ', '.join(words)} for tid, words in topic_words_list],
            "Model Response": {
                "Topic ID": model_response[0],
                "Words": ', '.join(model_response[1])
            },
            "Real Intruder": {
                "Topic ID": intruder_topic_id,
                "Words": ', '.join(intruder_topic_words)
            },
            "Intruder Index": answer_ix + 1  # Convert back to 1-based index
        }

        topic_intrusion_results.append(intrusion_result)

    # Optionally save for human evaluation
    if save_for_human_eval:
        with open("topic_intrusion_results.json", "w") as f:
            json.dump(topic_intrusion_results, f, indent=4)

    return topic_intrusion_results


# -------------------------
# Evaluate Model Performance
# -------------------------

def evaluate_word_intrusion_tasks(word_intrusion_results, TM_model):
    word_intrusion_correct = sum(1 for res in word_intrusion_results if res['Model Response'] == res['Real Intruder'])
    word_intrusion_accuracy = word_intrusion_correct / len(word_intrusion_results)

    print(f"Word Intrusion Task Accuracy: {word_intrusion_accuracy * 100:.2f}%")
    
    return {
        "Evaluated Model:": TM_model,
        "word_intrusion_accuracy": round(word_intrusion_accuracy, 5)
    }


def evaluate_topic_intrusion_tasks(topic_intrusion_results):
    topic_intrusion_correct = sum(1 for res in topic_intrusion_results if res['model_response'] == str(res['intruder_topic_id']))
    topic_intrusion_accuracy = topic_intrusion_correct / len(topic_intrusion_results)

    print(f"Topic Intrusion Task Accuracy: {topic_intrusion_accuracy * 100:.2f}%")
    
    return {        
        "Evaluated Model:": TM_model,
        "topic_intrusion_accuracy": topic_intrusion_accuracy
    }

# -------------------------
# Test LLM
# -------------------------

def test_llm_performance(words_lists, intruders):
    for intruder, word_list in zip(intruders, words_lists):
        
        numbered_word_list = ""
        for j, word in enumerate(word_list):
                numbered_word_list += f"{j + 1}. {word}\r\n"
        input_text = f"From the following list of tokens, identify the one token that does not belong with the others. For example: for\r\n1. Banana\r\n2. Orange\r\n3. Japan\r\n4. Strawberry\r\n5. Tree\r\nI expect the answer 3. Here are the words:\r\n {numbered_word_list}.\n In your response, provide only the intruder word's index without any additional explanation"
        messages = [
            {"role": "user", "content": f"{input_text}"},
        ]
        # Initialize pipeline with the loaded model

        # Generate output with constraints
        result = llm_model(
            messages,
            max_new_tokens=20,  # Limit to short responses
            no_repeat_ngram_size=2,
            return_full_text=False,  # Only return generated text
            temperature=temp,
        )

        clean_result = result[0]['generated_text'].strip()
        # index = int(clean_result) - 1
        # words_result = word_list[index]

        intrusion_result = {
            "prompt": input_text,
            "model_response": clean_result,
            "intruder": intruder
        }

        print("Words List: {}\nModel Intuder: {}\nReal Intruder: {}".format(word_list, clean_result, intruder))


def run_test():
    list_of_words = [['weapon', 'crime', 'rate', 'table', 'bill', 'license', 'control', 'carry', 'firearm'],
                    ['manager', 'window', 'problem', 'program', 'application', 'display', 'file', 'widget', 'blessing'],
                    ['peace', 'israel', 'israeli', 'taught', 'jewish', 'palestinian', 'muslim', 'arab', 'bosnia'],
                    ['medical', 'effect', 'slot', 'disease', 'cause', 'patient', 'treatment', 'food', 'doctor'],
                    ['hockey', 'team', 'baseball', 'game', 'information', 'season', 'player', 'play', 'league']]

    intruders = ['table', 'running', 'taught', 'slot', 'information']

    test_llm_performance(list_of_words, intruders)


# -------------------------
# Main Execution
# -------------------------

def main():
    for TM_model in TM_models:
        os.chdir("/home/dsi/ishonta/TopicDistributions")
        if not load_evaluation_from_file:
            dir_path = f"Intruders/{run_number}/{dataset}/{num_topics}/{TM_model}"  
            topic_word_df = load_csv(f"{dir_path}/10_intruder_check.csv")

            # Convert intruders pd to a list
            intruders = pd.read_csv(f"{dir_path}/the_intruders.csv")
            intruders = list(intruders.drop(intruders.columns[0], axis=1).values[0])

            # Run word intrusion task
            word_intrusion_results = updated_word_intrusion(topic_word_df, intruders=intruders, model=llm_model, TM_model=TM_model, save_for_human_eval=True)

        else:
            # Load results from file
            file_path = f"Evaluations/{run_number}/{dataset}/{num_topics}/{TM_model}_10_word_intrusion_results.json"
            with open(file_path, "r") as file:
                word_intrusion_results = json.load(file)

        # Evaluate results
        evaluation_results = evaluate_word_intrusion_tasks(word_intrusion_results, TM_model=TM_model)
        full_evaluation_results.append(evaluation_results)

    path_save_eval_result = f"Evaluations/{run_number}/{dataset}/{num_topics}"
    # Save evaluation results of all models to a file
    with open(f"{path_save_eval_result}/evaluation_results.json", "w") as f:
        json.dump(full_evaluation_results, f, indent=4)


# import argparse

# Initialize the argument parser
# parser = argparse.ArgumentParser(description="Run a model with the specified parameters.")
# parser.add_argument("--model_type", type=str, required=True, help="The type of model to run")
# parser.add_argument("--dataset", type=str, required=True, help="The dataset to use")

# args = parser.parse_args()
# TM_model = args.model_type
# dataset = args.dataset

if __name__ == "__main__":
    intruders = True
    load_evaluation_from_file = False
    full_evaluation_results = []
    path_save_eval_result = ""

    # # # # # # # # # # # # # # # # #
    # USE WHEN DONT RUN VIA SCRIPT  #
    # # # # # # # # # # # # # # # # #
    #                           
    if len(sys.argv) < 2:
        dataset = "20NewsGroup"
    else:
        dataset = sys.argv[1]
        num_topics = sys.argv[2]
        run_number = sys.argv[3]
        TM_models = sys.argv[4:]
        
    # run_test()
    main()
    

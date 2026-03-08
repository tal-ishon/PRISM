import pandas as pd
import numpy as np
import random
import sys
import os

# random.seed(0)

argv = sys.argv
if len(argv) > 2:
    DATASET = argv[1]
    NUM_TOPICS = argv[2]
    RUN_NUMBER = argv[3]
    MODELS = argv[4:]
    print("MODELS: {}".format(MODELS))
else:
    raise ValueError("Must provide a directory from which code load distributions")


def get_top_words_for_topics(topic_word_df, top_n=5):
    top_words_by_topic = {}
    for topic_id in topic_word_df['Topic']:
        # Get the row corresponding to the current topic (excluding the 'Topic' column)
        topic_row = topic_word_df.loc[topic_word_df['Topic'] == topic_id].drop(columns='Topic').iloc[0]
        # Sort the row values in descending order and get the top N columns (words)
        top_words = topic_row.sort_values(ascending=False).head(top_n)

        # Collect the top N words (column names)
        top_words_by_topic[topic_id] = top_words.index.tolist()

    return top_words_by_topic


def get_descending_sorted_words_per_topic(topic_word_df):
    sorted_words_by_topic = {}
    for topic_id in topic_word_df['Topic']:
        # Get the row corresponding to the current topic (excluding the 'Topic' column)
        topic_row = topic_word_df.loc[topic_word_df['Topic'] == topic_id].drop(columns='Topic').iloc[0]
        # Sort the row values in descending order
        top_words = topic_row.sort_values(ascending=False)

        # Collect the top N words (column names)
        sorted_words_by_topic[topic_id] = top_words.index.tolist()

    return sorted_words_by_topic


def get_GMM_top_words_per_topic(topic_word_df, top_n=5):
    top_words_by_topic = {}
    for topic_id in range(topic_word_df.shape[0]):
        # Get the row corresponding to the current topic (excluding the 'Topic' column)
        topic_row = topic_word_df.loc[topic_word_df['Topic'] == topic_id].drop(columns='Topic').iloc[0].dropna()
        # Sort the row values in descending order and get the top N columns (words)
        top_words = topic_row.head(top_n)

        # Collect the top N words (column names)
        top_words_by_topic[topic_id] = top_words.tolist()

    return top_words_by_topic


def get_sorted_GMM_words_per_topic(topic_word_df):
    top_words_by_topic = {}
    for topic_id in range(topic_word_df.shape[0]):
        # Get the row corresponding to the current topic (excluding the 'Topic' column)
        topic_row = topic_word_df.loc[topic_word_df['Topic'] == topic_id].drop(columns='Topic').iloc[0].dropna()

        # Collect the top N words (column names)
        top_words_by_topic[topic_id] = topic_row.tolist()

    return top_words_by_topic


def get_intruder_words(topic_word_df, GMM=False, index_begin_intruders=30):
    """
    Identify intruder words for each topic.
    - Finds low-probability words in each topic.
    - Checks if these words are among the top-k words in another topic.
    - If a low-probability word is not found in any top-k list, a random high-probability word from another topic is selected.
    """
    # Get top-k words for all topics
    if not GMM:
        sorted_words_by_topic = get_descending_sorted_words_per_topic(topic_word_df)
    else:
        sorted_words_by_topic = get_sorted_GMM_words_per_topic(topic_word_df)

    intruder_words = {}

    for topic_id in sorted_words_by_topic.keys():
        # Get the current topic row
        topic_row = sorted_words_by_topic[topic_id]

        # Find the lowest-probability words of that topic
        lowest_prob_words = topic_row[index_begin_intruders:]

        # Create a list of potential intruder words
        # All words that are in top 10 of other topics but in low probability of current topic
        num_top_words = 10
        potential_intruders = []
        for other_topic_id, sorted_words in sorted_words_by_topic.items():
            if other_topic_id != topic_id:
                potential_intruders.extend([word for word in sorted_words[:num_top_words] if word in lowest_prob_words])

        if potential_intruders:
            intruder_word = random.choice(potential_intruders)
            intruder_words[topic_id] = intruder_word
        else:
            print(f"No intruder found for topic {topic_id}")

    return intruder_words


def insert_intruders_into_top_words(topic_word_df, top_n_list=[5], GMM=False):
    """
    Insert intruder words into the top N words for each topic.
    """
    # Get max words list length in order to obtain from it all words lists.
    # We do so in order to have the same intruder for each top_n words in each topic.
    max_n = max(top_n_list)

    # Get the top words for each topic
    if not GMM:
        top_words_by_topic = get_top_words_for_topics(topic_word_df, top_n=max_n)
    else:
        top_words_by_topic = get_GMM_top_words_per_topic(topic_word_df, top_n=max_n)

    # Find the intruder words for each topic
    intruder_words = get_intruder_words(topic_word_df, GMM=GMM)

    # Insert the intruder word among the top words for each topic
    word_topic_list = []

    for top_n in top_n_list:
        updated_top_words_by_topic = {}

        for topic_id, top_words in top_words_by_topic.items():
            intruder = intruder_words[topic_id]
            if intruder not in top_words:
                # Insert the intruder word (ensure the list remains unique)
                updated_top_words = top_words[:top_n] + [intruder]
                random.shuffle(updated_top_words)
            else:
                # print in red letters as an error
                ValueError(f"Intruder word {intruder} already in top words for topic {topic_id}")

            updated_top_words_by_topic[topic_id] = updated_top_words

        word_topic_list.append(updated_top_words_by_topic)

    return word_topic_list, intruder_words

ks = [10]
for MODEL in MODELS:
    HOME_DIR = f"Distributions-Results/1/{DATASET}/{NUM_TOPICS}"
    
    INTRUDERS_PATH = f"Intruders/{RUN_NUMBER}/{DATASET}/{NUM_TOPICS}/{MODEL}"

    # Create directory if it doesn't exist
    if not os.path.exists(INTRUDERS_PATH):
        os.makedirs(INTRUDERS_PATH)
        
    df = pd.read_csv(f"{HOME_DIR}/{MODEL}_topic_word_distribution.csv")
    df = df.round(10)

    updated_top_words_by_topic, intruder_words = insert_intruders_into_top_words(df, top_n_list=ks)

    for i, top_words_topic in enumerate(updated_top_words_by_topic):
        updated_df = pd.DataFrame(top_words_topic)
        updated_df.to_csv(f"{INTRUDERS_PATH}/{ks[i]}_intruder_check.csv")

    intruders_df = pd.DataFrame(intruder_words, index=[0])
    intruders_df.to_csv(f"{INTRUDERS_PATH}/the_intruders.csv")


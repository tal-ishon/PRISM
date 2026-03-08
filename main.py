import argparse
import os
from contextlib import redirect_stdout

import pandas as pd
from ldamallet import LdaMallet

from utils import (
    load_octis_data,
    get_topics,
    compute_cv_coherence,
    calculate_npmi,
    compute_umass_coherence,
    compute_topic_diversity_coherence,
)


def save_distributions(model, model_type, dictionary, run_number, dataset, n_comp):
    path = f"TopicDistributions/Distributions-Results/{run_number}/{dataset}/{n_comp}"
    os.makedirs(path, exist_ok=True)

    topic_word_matrix = model.get_topics()
    topic_word_df = pd.DataFrame(
        topic_word_matrix,
        columns=[dictionary[i] for i in range(topic_word_matrix.shape[1])]
    )
    topic_word_df.to_csv(
        f"{path}/{model_type}_topic_word_distribution.csv",
        index_label="Topic"
    )


def print_topics(model, num_to_show=5, topn=10):
    available_topics = min(num_to_show, model.num_topics)
    for topic_id in range(available_topics):
        print(f"Topic {topic_id + 1}: \n{model.show_topic(topic_id, topn=topn)}")

    if model.num_topics < num_to_show:
        print(f"Only {model.num_topics} topics are available")


def main(
    n_comp,
    dataset,
    metric,
    num_iterations=1000,
    run_number="FINAL",
    seed=0,
    mallet_path="../mallet/bin/mallet",
):
    beta_value = f"mom_{metric}"

    dictionary, bow_corpus, doc_term_matrix, texts = load_octis_data(
        f"../data/{dataset}",
        to_save=False
    )

    lda = LdaMallet(
        mallet_path,
        doc_term_matrix,
        random_seed=seed,
        num_topics=n_comp,
        optimize_interval=10,
        iterations=num_iterations,
        id2word=dictionary,
        beta_path=f"priors/{dataset}/{n_comp}/{beta_value}.csv",
    )

    topics = get_topics(lda)

    log_file = f"{dataset.lower()}.txt"
    with open(log_file, "a", encoding="utf-8") as f:
        with redirect_stdout(f):
            print(f"\n###### Beta value: {beta_value} ######\n")
            print_topics(lda, num_to_show=5, topn=10)
            compute_umass_coherence("Mallet", lda)
            compute_cv_coherence("Mallet", topics)
            compute_topic_diversity_coherence("Mallet", lda, 10)
            calculate_npmi(topics, texts, "Mallet")

    print(f"\n###### Beta value: {beta_value} ######\n")
    print_topics(lda, num_to_show=5, topn=10)

    if run_number == "test":
        save_distributions(lda, "soc_mallet", dictionary, run_number, dataset, n_comp)
        cv = compute_cv_coherence(metric, topics)
        td = compute_topic_diversity_coherence(metric, lda, 10)
        npmi = calculate_npmi(topics, texts, metric)

        results_dir = f"Evaluations/results/{dataset}"
        os.makedirs(results_dir, exist_ok=True)

        with open(f"{results_dir}/PRISM.txt", "a", encoding="utf-8") as f:
            f.write(f"Number of Topics: {n_comp}\n")
            f.write(f"cv:{cv}\nTD:{td}\nnpmi:{npmi}\n\n")

    if metric == "glove":
        save_distributions(lda, "glove_mallet", dictionary, run_number, dataset, n_comp)

    if metric == "svd":
        save_distributions(lda, "svd_mallet", dictionary, run_number, dataset, n_comp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MALLET LDA with an estimated beta prior",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_mallet.py --n_comp 20 --dataset 20NewsGroup --metric prism
  python run_mallet.py --n_comp 25 --dataset BBC --metric 25_soc_pmi --num_iterations 2000
  python run_mallet.py --n_comp 20 --dataset DBLP --metric glove --run_number test --seed 42
        """,
    )

    parser.add_argument(
        "--n_comp",
        type=int,
        required=True,
        help="Number of topics/components for MALLET",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name",
    )
    parser.add_argument(
        "--metric",
        required=True,
        help="Metric name used to locate the estimated beta file",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=1000,
        help="Number of MALLET training iterations (default: 1000)",
    )
    parser.add_argument(
        "--run_number",
        default="FINAL",
        help="Run label used for saving outputs (default: FINAL)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--mallet_path",
        default="../mallet/bin/mallet",
        help="Path to the MALLET executable",
    )

    args = parser.parse_args()

    main(
        n_comp=args.n_comp,
        dataset=args.dataset,
        metric=args.metric,
        num_iterations=args.num_iterations,
        run_number=args.run_number,
        seed=args.seed,
        mallet_path=args.mallet_path,
    )
import argparse
import numpy as np
import graph_utils as gu


def main(
    n_comp,
    dataset,
    metrics="pmi",
    k=90,
    window_size=10,
    use_soc=False,
    use_glove=False,
    is_first=False,
    path_to_save=None,
):
    path_data = f"../data/{dataset}/{dataset}.json"
    corpus, dictionary = gu.load_data(path_data, type="json")

    origin_metric_name = metrics

    if is_first:
        pmi_matrix, vocab = gu.compute_ppmi_across_window_sizes(
            corpus=corpus,
            dataset=dataset,
            kind=metrics,
            is_first=is_first,
        )
    else:
        path_load = f"../data/{dataset}"
        pmi_matrix, vocab = gu.load_pmi_matrix(
            path=path_load,
            window_size=window_size,
        )

    if use_soc:
        print("soc matrix applied")
        affinity_matrix = gu.get_cosine_similarity(pmi_matrix)
        metric_name = f"{k}_soc_{origin_metric_name}"

    elif use_glove:
        print("glove matrix applied")
        affinity_matrix = gu.get_glove_matrix(
            dictionary,
            glove_path="glove/glove.6B.100d.txt",
            path_save="",
        )
        metric_name = "glove"

    else:
        affinity_matrix = pmi_matrix
        metric_name = f"{k}_{origin_metric_name}"

    gu.create_prior(
        affinity_matrix=affinity_matrix,
        corpus=corpus,
        vocab=vocab,
        dictionary=dictionary,
        dataset=dataset,
        w=None,
        n_components=n_comp,
        k=k,
        path_to_save=path_to_save,
        metric=metric_name,
        is_bayesian=True,
        smooth=True,
        eval_prior=True,
        is_first=is_first,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create word embeddings and priors for topic modeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_word_embeddings.py --n_comp 50 --dataset 20NewsGroup --metrics prism
  python create_word_embeddings.py --n_comp 5 --dataset BBC --metrics pmi --k 100 --use-soc
  python create_word_embeddings.py --n_comp 10 --dataset TrunpTweets --metrics prism --is-first --use-glove
        """,
    )

    parser.add_argument(
        "--n_comp",
        type=int,
        help="Number of components for dimensionality reduction",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (20NewsGroup, BBC, DBLP, M10, TrumpTweets)",
    )
    parser.add_argument(
        "--metric",
        default="pmi",
        help="Metric to use (default: pmi)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=90,
        help="Number of nearest neighbors (default: 90)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="Window size used when loading the PMI matrix",
    )
    parser.add_argument(
        "--is-first",
        action="store_true",
        help="Compute PMI matrix from scratch (first run)",
    )
    parser.add_argument(
        "--use-soc",
        action="store_true",
        help="Use second-order co-occurrence similarity matrix",
    )
    parser.add_argument(
        "--use-glove",
        action="store_true",
        help="Use GloVe word embeddings",
    )

    args = parser.parse_args()

    path_to_save = f"priors/{args.dataset}"

    main(
        n_comp=args.n_comp,
        dataset=args.dataset,
        metrics=args.metric,
        k=args.k,
        window_size=args.window_size,
        use_soc=args.use_soc,
        use_glove=args.use_glove,
        is_first=args.is_first,
        path_to_save=path_to_save,
    )
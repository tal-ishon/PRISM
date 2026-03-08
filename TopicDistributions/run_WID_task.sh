
# list of datasets

datasets=("20NewsGroup" "BBC" "M10" "DBLP" "TrumpTweets")
models=("mallet" "soc_mallet" "ProdLDA" "NeuralLDA" "BERTopic")
runs=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10")
# runs=("6" "7" "8" "9" "10")

# 20NewsGroup dataset is used for WID task
ng_topics=("20" "25" "50")

for topic in "${ng_topics[@]}"; do
    # create intruders
    for run in "${runs[@]}"; do
        python word_intrusion_creator.py \
        "${datasets[0]}" \
        "$topic" \
        "$run" \
        "${models[@]}"
    done

    # run WID task
    for run in "${runs[@]}"; do
        python wordIntrusionHF.py \
        "${datasets[0]}" \
        "$topic" \
        "$run" \
        "${models[@]}"
    done

done

# BBC dataset is used for WID task
bbc_topics=("5" "10" "15")

echo "Starting Word Intrusion Detection task for "${datasets[1]}" dataset"
for topic in "${bbc_topics[@]}"; do

    # create intruders
    for run in "${runs[@]}"; do
        python word_intrusion_creator.py \
        "${datasets[1]}" \
        "$topic" \
        "$run" \
        "${models[@]}"
    done

    # run WID task
    for run in "${runs[@]}"; do
        python wordIntrusionHF.py \
        "${datasets[1]}" \
        "$topic" \
        "$run" \
        "${models[@]}"
    done

done

# M10 dataset is used for WID task
m10_topics=("10" "15" "20")

for topic in "${m10_topics[@]}"; do
    # create intruders
    for run in "${runs[@]}"; do
        python word_intrusion_creator.py \
        "${datasets[2]}" \
        "$topic" \
        "$run" \
        "${models[@]}"
    done

    # run WID task
    for run in "${runs[@]}"; do
        python wordIntrusionHF.py \
        "${datasets[2]}" \
        "$topic" \
        "$run" \
        "${models[@]}"
    done

done

# DBLP dataset is used for WID task
dblp_topics=("4" "10" "15")

for topic in "${dblp_topics[@]}"; do
    # create intruders
    for run in "${runs[@]}"; do
        python word_intrusion_creator.py \
        "${datasets[3]}" \
        "$topic" \
        "$run" \
        "${models[@]}"
    done

    # run WID task
    for run in "${runs[@]}"; do
        python wordIntrusionHF.py \
        "${datasets[3]}" \
        "$topic" \
        "$run" \
        "${models[@]}"
    done

done

# TrumpTweets dataset is used for WID task
trump_topics=("10" "15" "20")

for topic in "${trump_topics[@]}"; do
    # create intruders
    for run in "${runs[@]}"; do
        python word_intrusion_creator.py \
        "${datasets[4]}" \
        "$topic" \
        "$run" \
        "${models[@]}"
    done

    # run WID task
    for run in "${runs[@]}"; do
        python wordIntrusionHF.py \
        "${datasets[4]}" \
        "$topic" \
        "$run" \
        "${models[@]}"
    done

done

# Echo the completion of the script
echo "Word Intrusion Detection task completed for all datasets and topics."
# End of script
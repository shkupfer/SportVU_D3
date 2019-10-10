INFILES_PATH=$1
PROCESSED_FILES_PATH=$2
RESULTS_PATH=$3
MODELS_PATH=$4

for JSON_FNAME in $( ls -d $INFILES_PATH* )
do
    python3 ingest/parse_game.py $JSON_FNAME
done

python3 preprocess/make_possessions.py

python3 preprocess/simple_feature_extraction.py $PROCESSED_FILES_PATH/all_data.csv

python3 model_build/train_test_rf.py $PROCESSED_FILES_PATH/all_data.csv $RESULTS_PATH/test_results.csv $MODELS_PATH/model.pkl
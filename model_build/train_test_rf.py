import argparse
import pickle
import logging
import sys

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

ignore_cols = ['game_id', 'team_id', 'poss_id']


def train_test_rf(infile_name, mdl_outfile, target_col, results_csv):
    logger.info("Reading input .csv: %s" % infile_name)
    df = pd.read_csv(infile_name)
    df.fillna(-999, inplace=True)

    # x_train, x_test, y_train, y_test = train_test_split(df[feature_cols], df[target_col], test_size=0.3, random_state=0)

    feature_cols = [col for col in df.columns if col not in ignore_cols + [target_col]]
    logger.info("%s features: %s" % (len(feature_cols), ', '.join(feature_cols)))
    logger.info("Target: %s" % target_col)

    cv = 3
    param_grid = {'n_estimators': [50, 100, 500, 1000], 'min_samples_leaf': [20, 50, 100, 200, 500, 1000],
                  'max_features': [0.1, 0.25, 0.5, 0.75, 0.9], 'bootstrap': [True], 'n_jobs': [-1]}

    rf_reg = RandomForestRegressor()

    gs_rf = GridSearchCV(rf_reg, param_grid, scoring='neg_mean_squared_error', cv=cv, n_jobs=1, verbose=3)

    logger.info("Fitting grid search random forest")
    gs_rf.fit(df[feature_cols], df[target_col])

    best_model = gs_rf.best_estimator_
    logger.info("Best model: %s" % best_model)
    best_score = gs_rf.best_score_
    logger.info("Average MSE across %s cross-validation sets for best model: %s" % (cv, best_score))

    if results_csv:
        gs_results_df = pd.DataFrame(gs_rf.cv_results_).sort_values(by='mean_test_score', ascending=False)
        gs_results_df.to_csv(results_csv, index=False)

    # y_pred = rf_reg.predict(x_test)
    # test_mse = mean_squared_error(y_test, y_pred)
    # logger.info("MSE on test data: %s" % test_mse)
    logger.info("Feature importances:\n%s" % '\n'.join([f_name + ': ' + str(f_imp) for f_name, f_imp in sorted(zip(feature_cols, best_model.feature_importances_),
                                                                                                               key=lambda name_imp: name_imp[1], reverse=True)]))
    # x_test[target_col] = y_test
    # x_test['pred_' + target_col] = y_pred
    # x_test.to_csv(test_outfile, index=False, header=True)

    with open(mdl_outfile, 'wb') as mdl_out:
        pickle.dump(rf_reg, mdl_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('infile_name')
    parser.add_argument('mdl_outfile')
    parser.add_argument('--target', default='points')
    parser.add_argument('--results_csv')
    args = parser.parse_args()

    train_test_rf(args.infile_name, args.mdl_outfile, args.target, args.results_csv)

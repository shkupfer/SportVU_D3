import argparse
import pickle
import logging
import sys

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

feature_cols = ['quarter', 'def_team_fouls', 'off_live_ball_tov', 'off_def_rebound', 'off_made_shot', 'score_margin',
                'game_clock_ms', 'shot_clock_ms', 'ball_dist', 'ball_angle']

target_col = 'points'


def train_test_rf(infile_name, test_outfile, mdl_outfile):
    df = pd.read_csv(infile_name)
    df.fillna(-999, inplace=True)

    x_train, x_test, y_train, y_test = train_test_split(df[feature_cols], df[target_col], test_size=0.3, random_state=0)

    rf_reg = RandomForestRegressor()
    rf_reg.fit(x_train, y_train)

    y_pred = rf_reg.predict(x_test)
    test_mse = mean_squared_error(y_test, y_pred)
    logger.info("MSE on test data: %s" % test_mse)
    logger.info("Feature importances:\n%s" % '\n'.join([f_name + ': ' + str(f_imp) for f_name, f_imp in sorted(zip(feature_cols, rf_reg.feature_importances_),
                                                                                                               key=lambda name_imp: name_imp[1], reverse=True)]))
    x_test[target_col] = y_test
    x_test['pred_' + target_col] = y_pred
    x_test.to_csv(test_outfile, index=False, header=True)

    with open(mdl_outfile, 'wb') as mdl_out:
        pickle.dump(rf_reg, mdl_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('infile_name')
    parser.add_argument('test_outfile')
    parser.add_argument('mdl_outfile')
    args = parser.parse_args()

    train_test_rf(args.infile_name, args.test_outfile, args.mdl_outfile)

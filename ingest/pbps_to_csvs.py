import argparse
from nba_api.stats.endpoints import playbyplayv2
import os
import sys
import time
import logging
import random

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('game_ids', nargs='+', type=str)
    parser.add_argument('--output_dir', '-o', default=os.getcwd())
    parser.add_argument('--sleep', '-s', default=1)
    args = parser.parse_args()

    for gameid in args.game_ids:
        logger.info("Downloading play-by-play for game with ID: %s" % gameid)
        pbp_df = playbyplayv2.PlayByPlayV2(gameid).play_by_play.get_data_frame()
        pbp_df.to_csv(os.path.join(args.output_dir, '%s_df.csv' % gameid))
        time.sleep((args.sleep / 2) + args.sleep * random.random())

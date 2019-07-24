import argparse
from nba_api.stats.endpoints import playbyplayv2
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('game_ids')
    parser.add_argument('--output_dir', '-o', default=os.getcwd())
    args = parser.parse_args()

    for gameid in args.game_ids:
        pbp_df = playbyplayv2.PlayByPlayV2(gameid).play_by_play.get_data_frame()
        pbp_df.to_csv(os.path.join(args.output_dir, '%s_df.csv' % gameid))

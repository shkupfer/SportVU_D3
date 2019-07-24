import argparse
import json
import datetime
from pytz import timezone
import logging
import sys
import re

import django
from django.db import transaction

django.setup()

import numpy as np

from nbad3.models import Team, Game, Player, PlayerStatus, Event, Moment, Coords
from ingest.utils import insert_ball_objs, team_colors, pbp_keys_translate
from nba_api.stats.endpoints import playbyplayv2
import pickle
import pandas as pd

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()


@transaction.atomic
def load_pbp(game_id_str, csv_filepath):
    game_obj = Game.objects.get(id=int(game_id_str))

    pbp_df = pd.read_csv(csv_filepath)

    home_score_after = visitor_score_after = 0
    home_team_fouls_after = visitor_team_fouls_after = 0

    penalty_re_exp = "^.*\(P[0-9]\.T*([0-9]*|PN)\)"
    penalty_re = re.compile(penalty_re_exp)

    for event_pbp_dict in pbp_df.to_dict(orient='records'):
        event_obj_args = {key: event_pbp_dict[pbp_keys_translate[key]] for key in pbp_keys_translate}
        for field in ('person1_type', 'person2_type', 'person3_type', 'player1_id', 'player2_id', 'player3_id',
                      'player1_team_id', 'player2_team_id', 'player3_team_id'):
            if np.isnan(event_obj_args[field]) or int(event_obj_args[field]) == 0:
                event_obj_args[field] = None
            else:
                event_obj_args[field] = int(event_obj_args[field])

        if event_pbp_dict.get('SCORE'):
            visitor_score_after, home_score_after = event_pbp_dict['SCORE'].split(' - ')
        event_obj_args['home_score_after'], event_obj_args['visitor_score_after'] = int(home_score_after), int(visitor_score_after)

        if event_obj_args['home_desc'] is not None:
            re_res = penalty_re.search(event_obj_args['home_desc'])
            if re_res:
                tf_from_pbp = re_res.group(1)
                if tf_from_pbp == 'PN':
                    home_team_fouls_after += 1
                else:
                    home_team_fouls_after = int(tf_from_pbp)
        if event_obj_args['visitor_desc'] is not None:
            re_res = penalty_re.search(event_obj_args['visitor_desc'])
            if re_res:
                tf_from_pbp = re_res.group(1)
                if tf_from_pbp == 'PN':
                    visitor_team_fouls_after += 1
                else:
                    visitor_team_fouls_after = int(tf_from_pbp)
        event_obj_args['home_team_fouls_after'], event_obj_args['visitor_team_fouls_after'] = int(home_team_fouls_after), int(visitor_team_fouls_after)

        event_obj_args['ev_real_time'] = datetime.datetime.strptime(event_pbp_dict['WCTIMESTRING'], "%H:%M %p").time()
        ev_gc_mins, ev_gc_secs = event_pbp_dict['PCTIMESTRING'].split(':')
        event_obj_args['ev_game_clock'] = datetime.timedelta(minutes=int(ev_gc_mins), seconds=int(ev_gc_secs))

        Player.objects.bulk_create([Player(id=event_obj_args['player%s_id' % pn]) for pn in range(1, 4) if
                                    event_obj_args['player%s_id' % pn] is not None], ignore_conflicts=True)

        event_obj_args['game'] = game_obj

        event_obj = Event.objects.create(**event_obj_args)
        logger.info(event_obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('game_id_str')
    parser.add_argument('csv_filepath')
    args = parser.parse_args()

    load_pbp(args.game_id_str, args.csv_filepath)

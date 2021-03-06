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

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()


@transaction.atomic
def parse_json(input_json):
    logger.info("Loading file: %s" % input_json)
    with open(input_json, 'r') as gamefile:
        game = json.load(gamefile)

    _, _, ball_player_status = insert_ball_objs()

    game_id_int = int(game['gameid'])
    game_date = datetime.datetime.strptime(game['gamedate'], '%Y-%m-%d').date()

    first_event = game['events'][0]

    home, visitor = first_event['home'], first_event['visitor']

    home_team_obj = Team(id=home['teamid'], name=home['name'], abbreviation=home['abbreviation'],
                         primary_color=team_colors[home['abbreviation']]['primary'], secondary_color=team_colors[home['abbreviation']]['secondary'])
    visitor_team_obj = Team(id=visitor['teamid'], name=visitor['name'], abbreviation=visitor['abbreviation'],
                            primary_color=team_colors[visitor['abbreviation']]['primary'], secondary_color=team_colors[visitor['abbreviation']]['secondary'])
    home_team_obj, visitor_team_obj = Team.objects.bulk_create([home_team_obj, visitor_team_obj], ignore_conflicts=True)
    logger.debug("Teams: %s" % [home_team_obj, visitor_team_obj])

    game_obj, _created = Game.objects.get_or_create(id=game_id_int, game_date=game_date, home=home_team_obj, visitor=visitor_team_obj)
    logger.info("Game: %s" % game_obj)

    current_quarter = 1

    for ind, event in enumerate(game['events']):
        # if len(event['moments']) == 0:
        #     logger.info("No moments in event #%s! Skipping" % ind)
        #     continue
        logger.info("Loading SportVU data for event #%s..." % (ind + 1))

        home, visitor = event['home'], event['visitor']
        player_dicts = [{'id': player['playerid'], 'first_name': player['firstname'], 'last_name': player['lastname']} for player in home['players'] + visitor['players']]
        for p_dict in player_dicts:
            Player.objects.update_or_create(**p_dict)

        home_goc_res = [PlayerStatus.objects.get_or_create(player_id=player['playerid'], team_id=home['teamid'], jersey=player['jersey'], position=player['position']) for player in home['players']]
        home_player_statuses, _ = zip(*home_goc_res)
        visitor_goc_res = [PlayerStatus.objects.get_or_create(player_id=player['playerid'], team_id=visitor['teamid'], jersey=player['jersey'], position=player['position']) for player in visitor['players']]
        visitor_player_statuses, _ = zip(*visitor_goc_res)

        if event['moments']:
            current_quarter = event['moments'][0][0]

        for mom_ind, moment in enumerate(event['moments']):
            logger.debug("Moment %s" % mom_ind)
            real_timestamp_seconds = datetime.datetime.fromtimestamp(int(str(moment[1])[:-3]), tz=timezone('UTC'))
            real_timestamp_milliseconds = datetime.timedelta(milliseconds=int(str(moment[1])[-3:]))
            real_timestamp = real_timestamp_seconds + real_timestamp_milliseconds
            game_clock = datetime.timedelta(seconds=moment[2])
            if moment[3] is not None:
                shot_clock = datetime.timedelta(seconds=moment[3])
            else:
                shot_clock = None
            moment_obj = Moment.objects.create(game=game_obj, quarter=current_quarter, real_timestamp=real_timestamp, game_clock=game_clock, shot_clock=shot_clock)

            coordsets = moment[5]

            coords_to_insert = []
            for coordset in coordsets:
                team_id, player_id, x, y, z = coordset

                player_status = [p_st for p_st in home_player_statuses + visitor_player_statuses + (ball_player_status, ) if p_st.player_id == player_id and p_st.team_id == team_id][0]
                logger.debug(player_status)

                coords_to_insert.append(Coords(player_status=player_status, x=x, y=y, z=z, moment=moment_obj))

            Coords.objects.bulk_create(coords_to_insert)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_json')
    args = parser.parse_args()

    parse_json(args.input_json)

import argparse
import csv
import logging
import sys

import django
from django.db.models import Q, F
from django.db.models.functions import ATan2, Extract

django.setup()

import numpy as np

from nbad3.models import Game, Moment, Possession

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

# (x, y) coordinates of the center of the rim at each side of the court
# Should I try this with (x, y, z)?
left_basket_coords = {'x': 4.75, 'y': 25}
right_basket_coords = {'x': 89.25, 'y': 25}

scaler = 180 / np.pi

# Find out which basket each team shoots at in OT
team_quarter_basket = {('home', 1): left_basket_coords, ('home', 2): left_basket_coords,
                       ('home', 3): right_basket_coords, ('home', 4): right_basket_coords,
                       ('visitor', 1): right_basket_coords, ('visitor', 2): right_basket_coords,
                       ('visitor', 3): left_basket_coords, ('visitor', 4): left_basket_coords}

output_cols = ['quarter', 'def_team_fouls', 'off_live_ball_tov', 'off_def_rebound', 'off_made_shot', 'score_margin',
               'game_clock_ms', 'shot_clock_ms', 'ball_dist', 'ball_angle', 'points']

def write_features(outfile_name):
    with open(outfile_name, 'w') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=output_cols)
        writer.writeheader()

        for game in Game.objects.all():
            home = game.home
            visitor = game.visitor
            for possession in Possession.objects.filter(game=game):
                logger.info(possession.long_desc())

                poss_attrs = {'quarter': possession.start_event.period,
                              'def_team_fouls': possession.start_event.visitor_team_fouls_after if possession.team == home else possession.start_event.home_team_fouls_after,
                              'off_live_ball_tov': 1 if possession.start_event.msg_type == 5 and possession.start_event.msg_action_type in (1, 2) else 0,  # Are these the only live ball turnover msg_action_types?
                              'off_def_rebound': 1 if possession.start_event.msg_type == 4 else 0,  # How do team rebounds work? If bouncing out of bounds counts as a team REB those should be excluded
                              'off_made_shot': 1 if possession.start_event.msg_type in (1, 3) else 0,  # Made shots (or FTs)
                              'score_margin': possession.start_event.home_score_after - possession.start_event.visitor_score_after if possession.team == home else possession.start_event.visitor_score_after - possession.start_event.home_score_after,  # Offensive team minus defensive team
                              'points': possession.points
                              }

                poss_moments = Moment.objects.filter(Q(game=game)
                                                     & Q(quarter=possession.start_event.period)
                                                     & Q(game_clock__lte=possession.start_event.ev_game_clock)
                                                     & Q(game_clock__gte=possession.end_event.ev_game_clock)).distinct('real_timestamp', 'quarter', 'game_clock', 'shot_clock')
                logger.info("Iterating through %s moments in this possession" % poss_moments.count())

                # poss_mom_coords_data = (poss_moments.annotate(x=ArrayAgg('coords__x', ordering='coords__player_status_id'),
                #                                               y=ArrayAgg('coords__y', ordering='coords__player_status_id'),
                #                                               z=ArrayAgg('coords__z', ordering='coords__player_status_id'),
                #                                               team_abbrev=ArrayAgg('coords__player_status__team__abbreviation',
                #                                                                    ordering='coords__player_status_id'),
                #                                               player=ArrayAgg('coords__player_status__player',
                #                                                               ordering='coords__player_status_id'))
                #                         .values('id', 'x', 'y', 'z', 'game_clock', 'shot_clock', 'team_abbrev', 'player')
                #                         .order_by('-game_clock', 'id')
                #                         )

                shooting_basket = team_quarter_basket[('home' if possession.team == home else 'visitor', possession.start_event.period)]

                ball_xy_data = (poss_moments.filter(coords__player_status__player=-1)
                                            .values(game_clock_ms=1000 * Extract('game_clock', 'epoch'), shot_clock_ms=1000 * Extract('shot_clock', 'epoch'),
                                            # .values('game_clock', 'shot_clock',
                                                    ball_dist=((F('coords__x') - shooting_basket['x']) ** 2 + (F('coords__y') - shooting_basket['y']) ** 2) ** 0.5,  # Distance from shooting basket
                                                    ball_angle=scaler * ATan2(F('coords__y') - shooting_basket['y'] if shooting_basket == left_basket_coords else shooting_basket['y'] - F('coords__y'),
                                                                              F('coords__x') - shooting_basket['x'] if shooting_basket == left_basket_coords else shooting_basket['x'] - F('coords__x')))  # Angle between basket and ball (-90 in right corner, 0 at top of key, 90 in left corner)
                                            # .order_by('-game_clock')
                                )

                for ball_dist_angle in ball_xy_data:
                    poss_attrs.update(ball_dist_angle)
                    writer.writerow(poss_attrs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('outfile_name')
    args = parser.parse_args()

    write_features(args.outfile_name)

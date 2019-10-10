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
from ingest.utils import ball_team

import random
from datetime import timedelta
from itertools import combinations
from operator import ge, le


logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

# (x, y) coordinates of the center of the rim at each side of the court
# Should I try this with (x, y, z)?
left_basket_coords = {'x': 4.75, 'y': 25}
right_basket_coords = {'x': 89.25, 'y': 25}

scaler = 180 / np.pi

# TODO: find out which basket each team shoots at in OT
team_quarter_basket = {('home', 1): left_basket_coords, ('home', 2): left_basket_coords,
                       ('home', 3): right_basket_coords, ('home', 4): right_basket_coords,
                       ('visitor', 1): right_basket_coords, ('visitor', 2): right_basket_coords,
                       ('visitor', 3): left_basket_coords, ('visitor', 4): left_basket_coords}


def xy_dist_btwn_coords(c1, c2):
    return np.sqrt((c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2)


def min_avg_max(arr):
    return np.min(arr), np.mean(arr), np.max(arr)


def centroid(coords_arr):
    x = np.mean([crd.x for crd in coords_arr])
    y = np.mean([crd.y for crd in coords_arr])
    return {'x': x, 'y': y}


def coords_basket_polar(coords, basket):
    r = np.sqrt((coords['x'] - basket['x']) ** 2 + (coords['y'] - basket['y']) ** 2)
    x1 = coords['y'] - basket['y'] if basket == left_basket_coords else basket['y'] - coords['y']
    x2 = coords['x'] - basket['x'] if basket == left_basket_coords else basket['x'] - coords['x']
    theta = scaler * np.arctan2(x1, x2)
    return {'r': r, 'theta': theta}


output_cols = ['quarter', 'def_team_fouls', 'off_live_ball_tov', 'off_def_rebound', 'off_made_shot', 'score_margin',
               'ball_r', 'game_clock', 'off_ball_min_dist', 'def_cent_r', 'def_ball_avg_dist', 'def_def_max_dist',
               'off_ball_avg_dist', 'off_ball_max_dist', 'off_off_min_dist', 'off_off_max_dist', 'shot_clock',
               'off_cent_r', 'def_ball_max_dist', 'ball_theta', 'off_cent_theta', 'def_ball_min_dist',
               'def_def_avg_dist', 'def_def_min_dist', 'def_cent_theta', 'off_off_avg_dist',
               'points']


def write_features(outfile_name, shot_clock_bounds, secs_before_end_bounds, secs_after_start_bounds, after_start_half_court):
    with open(outfile_name, 'w') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=output_cols)
        writer.writeheader()

        for game in Game.objects.all():
            home = game.home
            visitor = game.visitor
            for possession in Possession.objects.filter(game=game, valid=True):
                logger.info(possession.long_desc())
                offensive_team = possession.team
                defensive_team = home if offensive_team == visitor else visitor

                poss_attrs = {'quarter': possession.start_event.period,
                              'def_team_fouls': possession.start_event.visitor_team_fouls_after if possession.team == home else possession.start_event.home_team_fouls_after,
                              'off_live_ball_tov': 1 if possession.start_event.msg_type == 5 and possession.start_event.msg_action_type in (1, 2) else 0,  # Are these the only live ball turnover msg_action_types?
                              'off_def_rebound': 1 if possession.start_event.msg_type == 4 else 0,  # How do team rebounds work? If bouncing out of bounds counts as a team REB those should be excluded
                              'off_made_shot': 1 if possession.start_event.msg_type in (1, 3) else 0,  # Made shots (or FTs)
                              'score_margin': possession.start_event.home_score_after - possession.start_event.visitor_score_after if possession.team == home else possession.start_event.visitor_score_after - possession.start_event.home_score_after,  # Offensive team minus defensive team
                              'points': possession.points,
                              'game_id': game.id,
                              'team_id': possession.team.id,
                              'poss_id': possession.id
                              }

                poss_moments = Moment.objects.filter(Q(game=game)
                                                     & Q(quarter=possession.start_event.period)
                                                     & Q(game_clock__lte=possession.start_event.ev_game_clock)
                                                     & Q(game_clock__gte=possession.end_event.ev_game_clock)).distinct('real_timestamp', 'quarter', 'game_clock', 'shot_clock')

                shooting_basket = team_quarter_basket[('home' if possession.team == home else 'visitor', possession.start_event.period)]
                if shooting_basket == left_basket_coords:
                    half_court_op = le
                else:
                    half_court_op = ge

                if shot_clock_bounds:
                    poss_moments = poss_moments.filter(shot_clock__gte=timedelta(seconds=min(shot_clock_bounds)),
                                                       shot_clock__lte=timedelta(seconds=max(shot_clock_bounds)))
                if secs_before_end_bounds:
                    end_gc = possession.end_event.ev_game_clock
                    gc_bounds = [end_gc + timedelta(seconds=secs_before_end_bounds[0]), end_gc + timedelta(seconds=secs_before_end_bounds[1])]
                    poss_moments = poss_moments.filter(game_clock__gte=min(gc_bounds),
                                                       game_clock__lte=max(gc_bounds))
                if secs_after_start_bounds:
                    start_gc = possession.start_event.ev_game_clock
                    gc_bounds = [start_gc - timedelta(seconds=secs_after_start_bounds[0]), start_gc - timedelta(seconds=secs_after_start_bounds[1])]
                    poss_moments = poss_moments.filter(game_clock__gte=min(gc_bounds),
                                                       game_clock__lte=max(gc_bounds))

                if len(poss_moments) == 0:
                    logger.info("No moments matching filters for this possession")
                    continue

                logger.info("Iterating through %s moments in this possession" % poss_moments.count())
                past_half_court_gc = None
                for moment in poss_moments:
                    coords = moment.coords_set.all()

                    if after_start_half_court is not None:
                        if past_half_court_gc is not None and moment.game_clock < past_half_court_gc - timedelta(seconds=after_start_half_court):
                            pass
                        elif all([half_court_op(crd.x, 47) for crd in coords]):
                            past_half_court_gc = moment.game_clock
                            continue
                        else:
                            continue

                    game_clock = moment.game_clock.total_seconds()
                    shot_clock = moment.shot_clock.total_seconds() if moment.shot_clock is not None else -1

                    coord_pairs = combinations(coords, 2)

                    off_off_dists = []
                    def_def_dists = []
                    off_ball_dists = []
                    def_ball_dists = []

                    for c_pair in coord_pairs:
                        if c_pair[0].player_status.team == c_pair[1].player_status.team == offensive_team:
                            dist = xy_dist_btwn_coords(*c_pair)
                            off_off_dists.append(dist)
                        elif c_pair[0].player_status.team == c_pair[1].player_status.team == defensive_team:
                            dist = xy_dist_btwn_coords(*c_pair)
                            def_def_dists.append(dist)
                        elif ball_team in (c_pair[0].player_status.team, c_pair[1].player_status.team):
                            dist = xy_dist_btwn_coords(*c_pair)
                            if offensive_team in (c_pair[0].player_status.team, c_pair[1].player_status.team):
                                off_ball_dists.append(dist)
                            else:
                                def_ball_dists.append(dist)

                    off_off_min_dist, off_off_avg_dist, off_off_max_dist = min_avg_max(off_off_dists)
                    def_def_min_dist, def_def_avg_dist, def_def_max_dist = min_avg_max(def_def_dists)
                    off_ball_min_dist, off_ball_avg_dist, off_ball_max_dist = min_avg_max(off_ball_dists)
                    def_ball_min_dist, def_ball_avg_dist, def_ball_max_dist = min_avg_max(def_ball_dists)

                    off_cent_xy = centroid([crd for crd in coords if crd.player_status.team == offensive_team])
                    def_cent_xy = centroid([crd for crd in coords if crd.player_status.team == defensive_team])

                    off_cent_polar = coords_basket_polar(off_cent_xy, shooting_basket)
                    def_cent_polar = coords_basket_polar(def_cent_xy, shooting_basket)

                    ball_coords = coords.get(player_status__team=ball_team)
                    ball_polar = coords_basket_polar({'x': ball_coords.x, 'y': ball_coords.y}, shooting_basket)

                    moment_attrs = {'game_clock': game_clock, 'shot_clock': shot_clock,
                                    'off_off_min_dist': off_off_min_dist, 'off_off_avg_dist': off_off_avg_dist, 'off_off_max_dist': off_off_max_dist,
                                    'def_def_min_dist': def_def_min_dist, 'def_def_avg_dist': def_def_avg_dist, 'def_def_max_dist': def_def_max_dist,
                                    'off_ball_min_dist': off_ball_min_dist, 'off_ball_avg_dist': off_ball_avg_dist, 'off_ball_max_dist': off_ball_max_dist,
                                    'def_ball_min_dist': def_ball_min_dist, 'def_ball_avg_dist': def_ball_avg_dist, 'def_ball_max_dist': def_ball_max_dist,
                                    'off_cent_r': off_cent_polar['r'], 'off_cent_theta': off_cent_polar['theta'], 'def_cent_r': def_cent_polar['r'], 'def_cent_theta': def_cent_polar['theta'],
                                    'ball_r': ball_polar['r'], 'ball_theta': ball_polar['theta']}

                    poss_attrs.update(moment_attrs)
                    writer.writerow(poss_attrs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('outfile_name')

    parser.add_argument('--shot_clock_bounds', nargs=2, type=int)
    parser.add_argument('--secs_before_end_bounds', nargs=2, type=int)
    parser.add_argument('--secs_after_start_bounds', nargs=2, type=int)
    parser.add_argument('--after_start_half_court', type=int)

    args = parser.parse_args()

    write_features(args.outfile_name, args.shot_clock_bounds, args.secs_before_end_bounds, args.secs_after_start_bounds,
                   args.after_start_half_court)

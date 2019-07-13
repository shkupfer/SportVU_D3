import argparse
import logging
import sys

import django
from django.db import transaction

django.setup()

from nbad3.models import Team, Game, Player, PlayerStatus, Event, Coords, Moment, Possession

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()


@transaction.atomic
def make_possessions(game_id_int):
    game = Game.objects.get(id=game_id_int)
    this_game_events = Event.objects.filter(game=game)

    jump_balls = this_game_events.filter(msg_type=10)
    opening_tip = jump_balls.filter(pc_timestring="12:00").order_by('eventnum').last()
    logger.info("Opening tip event: %s" % opening_tip)
    team_won_tip = team_in_poss = opening_tip.player3_team
    prev_score = curr_score = {game.home: 0, game.visitor: 0}
    poss_over_flag = True
    made_shot_flag = False
    and1_ft_coming_flag = False
    home_score = visitor_score = home_team_fouls = visitor_team_fouls = 0

    for ind, event in enumerate(this_game_events[opening_tip.eventnum:]):
        logger.info("Event: %s" % event)

        # Keep track of the score so it can be differenced to determine the number of points scored on a possession
        prev_score = curr_score.copy()
        if event.home_score is not None and event.visitor_score is not None:
            curr_score = {game.home: event.home_score, game.visitor: event.visitor_score}

        # Don't save a made shot possession until we know it's not an and-1.
        # If it is an and-1, wait until we see the results of the FT before saving
        if made_shot_flag:
            if event.msg_type == 6 and event.msg_action_type in (2, 29) and event.player2_team == made_shot_poss_args['team']:
                and1_ft_coming_flag = True
                made_shot_flag = False
            else:
                poss_obj = Possession.objects.create(**made_shot_poss_args)
                logger.info("Created possession object: %s" % poss_obj.long_desc())
                poss_over_flag = True

        if poss_over_flag:
            poss_over_flag = False
            made_shot_flag = False
            and1_ft_coming_flag = False
            poss_points = 0
            poss_args = {'game': game, 'team': team_in_poss, 'start_event': opening_tip if ind == 0 else poss_obj.end_event,
                         'home_team_fouls_before': curr_score[game.home], 'visitor_team_fouls_before': curr_score[game.visitor],
                         'points': poss_points}

        # Jump balls
        if event.msg_type == 10 and event.player3_team != team_in_poss:
            poss_args.update({'end_event': event})
            poss_over_flag = True
            team_in_poss = event.player3_team

        # Defensive rebounds
        elif event.msg_type == 4 and event.msg_action_type == 0:
            if event.person1_type in (2, 4):
                reb_team = game.home
            elif event.person1_type in (3, 5):
                reb_team = game.visitor

            # If home team had the ball and visiting team got the rebound
            if team_in_poss == game.home and reb_team == game.visitor:
                poss_args.update({'end_event': event})
                poss_over_flag = True

            # If visiting team had the ball and home team got the rebound
            elif team_in_poss == game.visitor and reb_team == game.home:
                poss_args.update({'end_event': event})
                poss_over_flag = True

            team_in_poss = reb_team

        # Turnovers
        elif event.msg_type == 5:
            poss_args.update({'end_event': event})
            poss_over_flag = True
            team_in_poss = game.home if event.player1_team == game.visitor else game.visitor

        # Start of 2nd, 3rd, or 4th quarters
        elif event.msg_type == 12 and event.period > 1:
            poss_args.update({'end_event': event})
            poss_over_flag = True

            if event.period in (2, 3):
                team_in_poss = game.home if team_won_tip == game.visitor else game.visitor
            else:
                team_in_poss = team_won_tip

        # Made field goals
        elif event.msg_type == 1:
            made_shot_flag = True
            # If the home team made the shot
            if event.player1_team == game.home:
                poss_points += curr_score[game.home] - prev_score[game.home]
                team_in_poss = game.visitor
            # If the visiting team made the shot
            elif event.player1_team == game.visitor:
                poss_points += curr_score[game.visitor] - prev_score[game.visitor]
                team_in_poss = game.home

            # Create this possession object, but wait until we know whether it's an and-1 (and if so, the result of the FT) before saving
            made_shot_poss_args = {'game': game, 'team': event.player1_team, 'start_event': poss_args['start_event'],
                                   'end_event': event, 'points': poss_points}


        # Made free throws
        elif event.msg_type == 3 and event.msg_action_type in (10, 11, 12, 13, 14, 15) and (event.home_score is not None or event.visitor_score is not None):
            # If the home team made the FT
            if event.player1_team == game.home:
                poss_points += curr_score[game.home] - prev_score[game.home]
                if event.msg_action_type in (10, 12, 15):
                    team_in_poss = game.visitor
                    poss_over_flag = True
            # If the visiting team made the FT
            elif event.player1_team == game.visitor:
                poss_points += curr_score[game.visitor] - prev_score[game.visitor]
                if event.msg_action_type in (10, 12, 15):
                    team_in_poss = game.home
                    poss_over_flag = True

            if and1_ft_coming_flag is True:
                made_shot_poss_args['points'] = poss_points
                made_shot_poss_args['end_event'] = event
                poss_args = made_shot_poss_args
            else:
                poss_args.update({'end_event': event, 'points': poss_points})

        if poss_over_flag:
            poss_obj = Possession.objects.create(**poss_args)
            logger.info("Created possession object: %s" % poss_obj.long_desc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('game_id_int')
    args = parser.parse_args()

    game_obj = make_possessions(args.game_id_int)


import pickle
import django
from django.db.models import Q, Max, Min, Count, Case, When, IntegerField, FloatField
from django.db.models import F as dF
from django.db.models.functions import Cast, Round

django.setup()

from nbad3.models import Team, Game, Player, PlayerStatus, Event, Coords, Moment, Possession
from datetime import timedelta
import numpy as np
import torch
from ingest.utils import ball_team
import argparse
import logging
import os
import sys

FloatField.register_lookup(Round)

position_sort_order = ['', 'G', 'G-F', 'F-G', 'F', 'F-C', 'C-F', 'C']

bad_games = [21500022, 21500514, 21500009, 21500552, 21500620, 21500004, 21500034, 21500623, 21500584, 21500639, 21500645, 21500628, 21500053, 21500616, 21500506, 21500560, 21500597, 21500621, 21500084, 21500519, 21500544, 21500582, 21500647, 21500002, 21500049, 21500539, 21500029, 21500542, 21500625, 21500520, 21500060, 21500023, 21500016, 21500500, 21500521, 21500011, 21500043, 21500557, 21500073, 21500536, 21500572, 21500574, 21500660, 21500013, 21500545, 21500026, 21500564, 21500567, 21500634, 21500012, 21500032, 21500561, 21500062, 21500513, 21500490, 21500019, 21500530, 21500657, 21500044, 21500507, 21500538, 21500633, 21500031, 21500081, 21500629, 21500594, 21500027, 21500071, 21500566, 21500532, 21500653, 21500525, 21500510, 21500061, 21500017, 21500528, 21500041, 21500553, 21500001, 21500517, 21500556, 21500066, 21500496, 21500048, 21500565, 21500003, 21500495, 21500580, 21500632, 21500039, 21500056, 21500571, 21500546, 21500618, 21500036, 21500033, 21500522, 21500569, 21500585, 21500648, 21500491, 21500658, 21500577, 21500493, 21500636]

def get_data_targets(outfiles_dir, buffer_secs, max_secs, fade_timesteps, player_chans, one_faded_img):
    logger.info("Top of func")
    hc_possessions = Possession.objects.filter(valid=True, half_court=True)
    hc_possessions = hc_possessions.exclude(game__in=bad_games).order_by('id')
    n_poss = hc_possessions.count()
    for n, poss in enumerate(hc_possessions):
        if n % 10 == 0:
            logger.info("Processing possession %s of %s" % (n + 1, n_poss))

        home = poss.game.home
        game_clock_upper_bound = poss.start_event.ev_game_clock - timedelta(seconds=buffer_secs)

        buffered_event_end = poss.end_event.ev_game_clock + timedelta(seconds=buffer_secs)
        after_max_secs = poss.start_event.ev_game_clock - timedelta(seconds=max_secs)
        game_clock_lower_bound = max(buffered_event_end, after_max_secs)

        hc_moments = (Moment.objects.filter(Q(game=poss.game)
                                            & Q(quarter=poss.start_event.period)
                                            & Q(game_clock__lte=game_clock_upper_bound)
                                            & Q(game_clock__gte=game_clock_lower_bound))
                                    .distinct('real_timestamp', 'quarter', 'game_clock', 'shot_clock')).order_by('-game_clock')

        moms_data_secs = (game_clock_upper_bound - game_clock_lower_bound).total_seconds()
        # TODO: LOOK AT THIS!!!
        if len(hc_moments) < 25 * moms_data_secs + 1:
            logger.info("LESS THAN %s MOMENTS (%s) ON THIS POSSESSION, NOT WRITING IT:" % (25 * moms_data_secs + 1, len(hc_moments)))
            logger.info(poss.long_desc())
            continue
        hc_moments = hc_moments[:int(moms_data_secs * 25)]
        logger.info("Using %s moments from this possession" % len(hc_moments))

        team_ball_case_stmt = Case(When(player_status__team=poss.team, then=0),
                                   When(player_status__team=ball_team, then=1),
                                   default=2, output_field=IntegerField())

        pos_case_stmt = Case(*[When(player_status__position=position, then=ind) for ind, position in enumerate(position_sort_order)],
                             output_field=IntegerField())

        if player_chans:
            first_mom_coords = Coords.objects.filter(moment=hc_moments[0])
            player_chans_translate = dict((pid, ind) for ind, pid in enumerate(first_mom_coords.order_by(team_ball_case_stmt, pos_case_stmt).values_list('player_status__id', flat=True)))

            all_coords = [list(Coords.objects.filter(moment=mom).values(x_rnd=Cast('x__round', IntegerField()) - 47 if poss.going_to_right else 47 - Cast('x__round', IntegerField()),
                                                                        y_rnd=Cast('y__round', IntegerField()))
                                                                .values_list('player_status__id',
                                                                             Case(When(x_rnd__lt=0, then=0), When(x_rnd__gt=47, then=47), default='x_rnd'),
                                                                             Case(When(y_rnd__lt=0, then=0), When(y_rnd__gt=50, then=50), default='y_rnd'))
                               ) for mom in hc_moments]

            trans_ac = [list(zip(*ccc)) for ccc in all_coords]
            trans_ac_arr = np.array(trans_ac)
            old_chans = trans_ac_arr[:, 0, :]
            try:
                new_chans = [player_chans_translate[k] for k in old_chans.flat]
            except KeyError as exp:
                print("Key error, something weird happened with player status ID: %s" % str(exp))
                continue
            trans_ac_arr[:, 0, :] = np.array(new_chans).reshape((len(hc_moments), 11))
            trans_ac = trans_ac_arr.tolist()
            n_chans = 11
        else:
            all_coords = [list(Coords.objects.filter(moment=mom).values(tm=team_ball_case_stmt,
                                                                        x_rnd=Cast('x__round', IntegerField()) - 47 if poss.going_to_right else 47 - Cast('x__round', IntegerField()),
                                                                        y_rnd=Cast('y__round', IntegerField()))
                                                                .filter(tm__lte=1)
                                                                .values_list(dF('tm'),
                                                                             Case(When(x_rnd__lt=0, then=0), When(x_rnd__gt=47, then=47), default='x_rnd'),
                                                                             Case(When(y_rnd__lt=0, then=0), When(y_rnd__gt=50, then=50), default='y_rnd'))
                               ) for mom in hc_moments]

            trans_ac = [list(zip(*ccc)) for ccc in all_coords]
            n_chans = 2

        if fade_timesteps:
            fade_step = 1 / fade_timesteps
            poss_courts = []
            court = np.zeros((n_chans, 48, 51))
            for mom_coords in trans_ac:
                court = court - fade_step
                court = court * (court > 0)
                court[tuple(mom_coords)] = 1
                poss_courts.append(court)
        else:
            poss_courts = []
            for mom_coords in trans_ac:
                court = np.zeros((n_chans, 48, 51))
                court[tuple(mom_coords)] = 1
                poss_courts.append(court)

        if hc_moments[0].shot_clock is None:
            shot_clock = None
            shot_clock_ind = 0
            while shot_clock is None and shot_clock_ind < len(hc_moments):
                shot_clock = hc_moments[shot_clock_ind].shot_clock
                shot_clock_ind += 1
            if shot_clock is not None:
                shot_clock_val = shot_clock.total_seconds()
            else:
                continue
        else:
            shot_clock_val = hc_moments[0].shot_clock.total_seconds()


                     # Started with a made field goal, started with a made free throw, started with a new quarter (otherwise it started with a rebound)
        poss_data = [1 if poss.start_event.msg_type == 4 else 0, 1 if poss.start_event.msg_type == 3 else 0, 1 if poss.start_event.msg_type == 12 else 0,
                     poss.start_event.period, hc_moments[0].game_clock.total_seconds(), shot_clock_val,
                     poss.start_event.visitor_team_fouls_after if poss.team == home else poss.start_event.home_team_fouls_after]

        if one_faded_img:
            poss_courts = poss_courts[-1]
        outdict = {'data': torch.Tensor(poss_courts), 'target': torch.Tensor([poss.points]),
                   'poss_data': torch.Tensor(poss_data)}

        with open(os.path.join(outfiles_dir, '%s.pkl' % poss.id), 'wb') as pklfile:
            pickle.dump(outdict, pklfile)

    # all_poss_points = torch.Tensor(hc_possessions.values_list('points', flat=True))
    # all_poss_points = all_poss_points.reshape((all_poss_points.size(0), -1))
    # all_poss_courts = torch.Tensor(all_poss_courts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-buffer_secs', '-B', type=int)
    parser.add_argument('-max_secs', '-M', type=int)
    parser.add_argument('-fade_timesteps', '-F', type=int)
    parser.add_argument('-player_chans', '-P', action='store_true')
    parser.add_argument('-one_faded_img', '-I', action='store_true')
    parser.add_argument('-logfile', '-l')
    parser.add_argument('outfiles_dir')

    args = parser.parse_args()

    if args.logfile:
        logging.basicConfig(level=logging.INFO, filename=args.logfile)
    else:
        logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger()

    get_data_targets(args.outfiles_dir, args.buffer_secs, args.max_secs, args.fade_timesteps, args.player_chans, args.one_faded_img)

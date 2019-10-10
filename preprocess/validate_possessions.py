import django
from django.db import transaction
import logging
import sys
from django.db.models import Q, F
from django.db.models import Max, Min, Count
from datetime import timedelta
import argparse
from operator import ge, le

django.setup()

from nbad3.models import Team, Game, Player, PlayerStatus, Event, Coords, Moment, Possession

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()


left_basket_coords = {'x': 4.75, 'y': 25}
right_basket_coords = {'x': 89.25, 'y': 25}

team_quarter_basket = {('home', 1): left_basket_coords, ('home', 2): left_basket_coords,
                       ('home', 3): right_basket_coords, ('home', 4): right_basket_coords,
                       ('visitor', 1): right_basket_coords, ('visitor', 2): right_basket_coords,
                       ('visitor', 3): left_basket_coords, ('visitor', 4): left_basket_coords}


def validate_possessions(min_hc_secs, examine_hc_secs):
    for game in Game.objects.all().order_by('id'):
        with transaction.atomic():
            logger.info("Validating possessions for game: %s\n\n" % game)
            home = game.home
            visitor = game.visitor
            game_possessions = Possession.objects.filter(game=game).order_by('start_event__period', '-start_event__ev_game_clock')

            for poss in game_possessions:
                logger.info(poss.long_desc())
                ev_start = poss.start_event.ev_game_clock
                ev_end = poss.end_event.ev_game_clock
                poss_moments = (Moment.objects.filter(Q(game=game)
                                                      & Q(quarter=poss.start_event.period)
                                                      & Q(game_clock__lte=ev_start)
                                                      & Q(game_clock__gte=ev_end))
                                .distinct('real_timestamp', 'quarter', 'game_clock', 'shot_clock'))
                poss_moments = Moment.objects.filter(id__in=poss_moments)

                shooting_basket = team_quarter_basket.get(('home' if poss.team == home else 'visitor', poss.start_event.period), None)
                if shooting_basket == left_basket_coords:
                    poss.going_to_right = False
                    minmax_x = max
                    ge_le_x = le
                elif shooting_basket == right_basket_coords:
                    poss.going_to_right = True
                    minmax_x = min
                    ge_le_x = ge

                valid = True
                if not 0 <= poss.points <= 4:
                    print("** REJECTED** : Possession has %s points" % poss.points)
                    valid = False
                elif valid and len(poss_moments) == 0:
                    print("** REJECTED** : Has no moments")
                    valid = False
                else:
                    first_last_gc = poss_moments.aggregate(Min('game_clock'), Max('game_clock'))
                    # n_unique_gc = poss_moments.values('game_clock').distinct().count()
                    # n_moments = poss_moments.count()
                    # expect_unique_gc = 25 * (first_last_gc['game_clock__max'] - first_last_gc['game_clock__min']).total_seconds() + 1

                    start_end_makes_sense = first_last_gc['game_clock__max'].seconds + 1 >= ev_start.seconds and first_last_gc['game_clock__min'].seconds - 1 <= ev_end.seconds
                    if not start_end_makes_sense:
                        print("** REJECTED** : Start/end event times don't make sense with start/end moment times: Events: %s -> %s, moments: %s -> %s" % (ev_start, ev_end, first_last_gc['game_clock__max'], first_last_gc['game_clock__min']))
                        valid = False

                    # if valid and abs(n_moments - expect_unique_gc) > 5.0:
                    #     print("** REJECTED** : Number of moments is not what was expected: expected %s, got %s" % (expect_unique_gc, n_moments))
                    #     valid = False

                if valid:
                    # min_mom_coords = poss_moments.annotate(Count('coords')).aggregate(Min('coords__count'))['coords__count__min']
                    min_mom_coords = min([m.coords_set.count() for m in poss_moments])
                    if min_mom_coords != 11:
                        print("** REJECTED** : Has a moment with %s coords (all should have 11 coords)" % min_mom_coords)
                        valid = False

                if valid:
                    print("Possession is valid!")

                half_court = True
                first_time_hc = None
                # Including only possessions that are in regulation, come off of defensive rebounds or made shots,
                # start with over a minute on the game clock, and last at least min_hc_secs seconds after all players/ball cross half court
                if not valid:
                    half_court = False
                if half_court and poss.start_event.period > 4:
                    print("** REJECTED** : Possession is in overtime (period %s), not including in half court" % poss.start_event.period)
                    half_court = False
                if half_court and poss.start_event.msg_type not in (1, 3, 4, 12):
                    print("** REJECTED** : Possession does not start with a new quarter, defensive rebound or made shot (msg type %s), not including in half court" % poss.start_event.msg_type)
                    half_court = False
                if half_court and poss.start_event.ev_game_clock < timedelta(minutes=1):
                    print("** REJECTED** : Possession starts with less than 1 minute on game clock (%s), not including in half court" % poss.start_event.ev_game_clock)
                    half_court = False
                if half_court and (poss.start_event.ev_game_clock - poss.end_event.ev_game_clock).total_seconds() < min_hc_secs:
                    print("** REJECTED** : Only %s between start and end (events) of possession, so it cannot have the %s of half court time that is required to be included in half court" % ((poss.start_event.ev_game_clock - poss.end_event.ev_game_clock), min_hc_secs))
                    half_court = False
                if half_court:
                    # Look for the first time when all 11 players/ball were across half court
                    # first_time_hc = poss_moments.annotate(**annotate_kwargs).values('game_clock', list(annotate_kwargs.keys())[0]).filter(**filter_kwargs).values('game_clock', list(annotate_kwargs.keys())[0]).aggregate(Max('game_clock'))['game_clock__max']
                    gc_and_minmax_x = [(m.game_clock, minmax_x(m.coords_set.all().values_list('x', flat=True))) for m in poss_moments]
                    past_hc = [(gc, mx) for gc, mx in gc_and_minmax_x if ge_le_x(mx, 47)]
                    first_time_hc = max([gc for gc, mx in past_hc]) if len(past_hc) > 0 else None
                    if first_time_hc is None:
                        print("** REJECTED** : All coords never crossed half court, not including in half court")
                        half_court = False
                    if half_court and (first_time_hc - poss.end_event.ev_game_clock).total_seconds() < min_hc_secs:
                        print("** REJECTED** : Only %s between first time across half court and end (event) of possession, %s is required to be included in half court" % ((first_time_hc - poss.end_event.ev_game_clock).total_seconds(), min_hc_secs))
                        half_court = False
                    if half_court:
                        hc_moments_count = (poss_moments.filter(Q(game_clock__lte=first_time_hc)
                                                                & Q(game_clock__gte=first_time_hc - timedelta(seconds=examine_hc_secs + .04)))
                                                        .count())
                        if hc_moments_count < 25 * examine_hc_secs or hc_moments_count > 25 * examine_hc_secs + 5:
                            print("** REJECTED** : %s half court moments, should have %s" % (hc_moments_count, 25 * examine_hc_secs))
                            half_court = False

                if half_court:
                    print("This is a valid, half court possession. %s of half court time, starting at %s" % (first_time_hc - poss.end_event.ev_game_clock, first_time_hc))

                poss.valid = valid
                poss.half_court = half_court
                poss.hc_start = first_time_hc

                poss.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-min_hc_secs', '-N', type=int,
                        help='Number of seconds required after crossing half court in order to include possession')
    parser.add_argument('-examine_hc_secs', '-M', type=int, help='num')
    args = parser.parse_args()

    validate_possessions(args.min_hc_secs, args.examine_hc_secs)

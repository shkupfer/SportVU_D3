import django
from django.db import transaction
import logging
import sys
from django.db.models import Q, F
from django.db.models import Max, Min, Count

django.setup()

from nbad3.models import Team, Game, Player, PlayerStatus, Event, Coords, Moment, Possession

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()


@transaction.atomic
def validate_possessions():
    for game in Game.objects.all():
        logger.info("Validating possessions for game: %s" % game)
        game_possessions = Possession.objects.filter(game=game)

        for poss in game_possessions:
            ev_start = poss.start_event.ev_game_clock
            ev_end = poss.end_event.ev_game_clock
            poss_moments = (Moment.objects.filter(Q(game=game)
                                                  & Q(quarter=poss.start_event.period)
                                                  & Q(game_clock__lte=ev_start)
                                                  & Q(game_clock__gte=ev_end))
                            .distinct('real_timestamp', 'quarter', 'game_clock', 'shot_clock')).order_by('-game_clock')

            if len(poss_moments) == 0:
                valid = False
            else:
                first_last_gc = Moment.objects.filter(id__in=poss_moments).aggregate(Min('game_clock'), Max('game_clock'))
                n_unique_gc = poss_moments.values('game_clock').distinct().count()
                expect_unique_gc = 25 * (first_last_gc['game_clock__max'] - first_last_gc['game_clock__min']).total_seconds() + 1

                if first_last_gc['game_clock__max'].seconds + 1 >= ev_start.seconds and first_last_gc['game_clock__min'].seconds - 1 <= ev_end.seconds and n_unique_gc >= expect_unique_gc:
                    valid = True
                else:
                    valid = False

            if valid:
                min_mom_coords = Moment.objects.filter(id__in=poss_moments.values_list('id', flat=True)).annotate(Count('coords')).aggregate(Min('coords__count'))['coords__count__min']
                if min_mom_coords != 11:
                    valid = False

            poss.valid = valid
            poss.save()


if __name__ == "__main__":
    validate_possessions()

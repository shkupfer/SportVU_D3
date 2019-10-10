import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from django.db.models import Case, When, Value, IntegerField, Q, CharField
from django.db.models.functions import Concat, Cast, ExtractMinute, ExtractSecond, Floor, LPad
from django.contrib.postgres.aggregates import ArrayAgg
from .models import Moment, Event, Possession
import logging
from datetime import timedelta
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()


ball_radius = 7
player_radius = 12
examine_hc_secs = 6

def speed_from_coords(first_coords, second_coords, elapsed_secs=1. / 25, scale=3600. / 5280):
    distance = np.sqrt((first_coords['x'] - second_coords['x']) ** 2 + (first_coords['y'] - second_coords['y']) ** 2 + (first_coords['z'] - second_coords['z']) ** 2)
    return scale * (distance / elapsed_secs)


def play_anim_data(request, possession_id, half_court):
    logger.info("At top of play_anim_data. Request: %s" % str(request))
    poss = Possession.objects.get(id=possession_id)
    game = poss.game

    if half_court == 'on':
        anim_start = poss.hc_start
        anim_end = poss.hc_start - timedelta(seconds=examine_hc_secs + .04)
    else:
        anim_start = poss.start_event.ev_game_clock
        anim_end = poss.end_event.ev_game_clock

    moment_ids = (Moment.objects.filter(Q(game=game)
                                        & Q(quarter=poss.start_event.period)
                                        & Q(game_clock__lte=anim_start)
                                        & Q(game_clock__gte=anim_end))
                  .distinct('real_timestamp', 'quarter', 'game_clock', 'shot_clock').values_list('id', flat=True))
    logger.info("Got moment IDs for this possession")

    # TODO: Order by real timestamp rather than game clock? First need to change parsing script so it gets the milliseconds for real timestamp
    # TODO: Should make that parsing change anyway
    # TODO: There has to be a better way to do this, right?
    moments = Moment.objects.filter(id__in=moment_ids)
    logger.info("Got moments")
    data = (moments.annotate(x=ArrayAgg('coords__x', ordering='coords__player_status_id'), y=ArrayAgg('coords__y', ordering='coords__player_status_id'), z=ArrayAgg('coords__z', ordering='coords__player_status_id'),
                             radius=ArrayAgg(Case(When(coords__player_status__player_id=-1, then=Value(ball_radius)), default=Value(player_radius), output_field=IntegerField()), ordering='coords__player_status_id'),
                             jersey=ArrayAgg('coords__player_status__jersey', ordering='coords__player_status_id'),
                             team_abbrev=ArrayAgg('coords__player_status__team__abbreviation', ordering='coords__player_status_id'),
                             first_name=ArrayAgg('coords__player_status__player__first_name', ordering='coords__player_status_id'),
                             last_name=ArrayAgg('coords__player_status__player__last_name', ordering='coords__player_status_id'))
                   .values('id', 'x', 'y', 'z', 'radius', 'jersey', 'team_abbrev', 'first_name', 'last_name')
                   .order_by('-game_clock', 'id')
            )
    # logger.info("Queried annotated data, trying to print first 3")
    # logger.info(data[:3])

    logger.info("Getting clocks data")
    game_clock_times = [str(gc_td)[2:7] for gc_td in moments.values_list('game_clock', flat=True).order_by('-game_clock')]
    shot_clock_times = [str(sc_td)[4:7] for sc_td in moments.values_list('shot_clock', flat=True).order_by('-game_clock')]
    # game_shot_clock_times = moments.values(game_clock_fmt=ToChar('game_clock', Value('MI:SS')), shot_clock_fmt=ToChar('shot_clock', Value(':SS'))).order_by('-game_clock')
    # game_shot_clock_times = moments.values(game_clock_fmt=Concat(Cast(ExtractMinute('game_clock'), CharField()), Value(':'), LPad(Cast(Floor(ExtractSecond('game_clock')), CharField()), 2, Value('0'))),
    #                shot_clock_fmt=Concat(Value(':'), LPad(Cast(Floor(ExtractSecond('shot_clock')), CharField()), 2, Value('0')))).order_by('-game_clock')
    logger.info("Got clocks data")

    # TODO: Is there a better/more Pythonic way to do this?
    # TODO: Couldn't figure out how to do it with pure Django
    # speed_data = [11 * [0]]
    # logger.info("Calling list() on data")
    # data = list(data)
    logger.info("Starting rearranging loop")
    out_data = []
    for ind in range(len(data)):
        out_data.append({'marker_data': [{'x': x, 'y': y, 'z': z, 'radius': radius, 'jersey': jersey, 'team_abbrev': team_abbrev,
                                      'h_or_v': 'home' if team_abbrev == game.home.abbreviation else 'visitor',
                                      'name': first_name + ' ' + last_name}
                                      for x, y, z, radius, jersey, team_abbrev, first_name, last_name in
                                      zip(data[ind]['x'], data[ind]['y'], data[ind]['z'], data[ind]['radius'], data[ind]['jersey'], data[ind]['team_abbrev'], data[ind]['first_name'], data[ind]['last_name'])],
                     'game_clock': game_clock_times[ind], 'shot_clock': shot_clock_times[ind]})
    logger.info("Finished for loop rearranging data")
    logger.info(out_data[0])
        # if ind > 0:
        #     this_ind_speed = [speed_from_coords(cur, prev) for cur, prev in zip(data[ind]['marker_data'], data[ind - 1]['marker_data'])]
        #     speed_data.append(this_ind_speed)

    big_resp = {'moments_data': out_data, 'quarter': poss.start_event.period, 'home_score': poss.start_event.home_score_after,
                'visitor_score': poss.start_event.visitor_score_after, 'home_abbrev': game.home.abbreviation, 'visitor_abbrev': game.visitor.abbreviation}

    resp = JsonResponse(big_resp, safe=False)
    logger.info("About to return response")

    return resp


def coach(request):
    from .forms import PossessionSelector
    if request.method == 'POST':
        form = PossessionSelector()
    else:
        form = PossessionSelector()

    logger.info("At bottom of coach()")
    return render(request, 'html/coach.html', {'form': form})


def load_possessions(request):
    print(request.GET.dict())
    possessions = Possession.objects
    game_ids = request.GET.getlist('games')
    half_court = request.GET.get('half_court')
    print(game_ids)
    print(half_court)

    if game_ids and game_ids != ['']:
        possessions = possessions.filter(Q(game_id__in=game_ids))
    if half_court == 'on':
        possessions = possessions.filter(half_court=True)


    loaded_possessions = possessions.order_by('start_event__eventnum')
    return render(request, 'html/dropdown_lists/possession_options.html', {'possessions': loaded_possessions})

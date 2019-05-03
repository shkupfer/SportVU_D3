import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from django.db.models import Case, When, Value, IntegerField, Q
from django.contrib.postgres.aggregates import ArrayAgg
from .models import Moment, Event
from .forms import EventSelector

ball_radius = 7
player_radius = 12


def speed_from_coords(first_coords, second_coords, elapsed_secs=1. / 25, scale=3600. / 5280):
    distance = np.sqrt((first_coords['x'] - second_coords['x']) ** 2 + (first_coords['y'] - second_coords['y']) ** 2 + (first_coords['z'] - second_coords['z']) ** 2)
    return scale * (distance / elapsed_secs)


def play_anim_data(request, event_id):
    event = Event.objects.get(id=event_id)
    game = event.game
    moments = Moment.objects.filter(event=event)
    # TODO: Order by real timestamp rather than game clock? First need to change parsing script so it gets the milliseconds for real timestamp
    # TODO: Should make that parsing change anyway
    # TODO: There has to be a better way to do this, right?
    data = (moments.annotate(x=ArrayAgg('coords__x', ordering='coords__player_status_id'), y=ArrayAgg('coords__y', ordering='coords__player_status_id'), z=ArrayAgg('coords__z', ordering='coords__player_status_id'),
                             radius=ArrayAgg(Case(When(coords__player_status__player_id=-1, then=Value(ball_radius)), default=Value(player_radius), output_field=IntegerField()), ordering='coords__player_status_id'),
                             jersey=ArrayAgg('coords__player_status__jersey', ordering='coords__player_status_id'),
                             team_abbrev=ArrayAgg('coords__player_status__team__abbreviation', ordering='coords__player_status_id'),
                             first_name=ArrayAgg('coords__player_status__player__first_name', ordering='coords__player_status_id'),
                             last_name=ArrayAgg('coords__player_status__player__last_name', ordering='coords__player_status_id'))
                   .values('id', 'x', 'y', 'z', 'radius', 'jersey', 'team_abbrev', 'first_name', 'last_name')
                   .order_by('-game_clock', 'id')
            )
    data = list(data)

    game_clock_times = [str(gc_td)[2:7] for gc_td in moments.values_list('game_clock', flat=True).order_by('-game_clock')]
    shot_clock_times = [str(sc_td)[4:7] for sc_td in moments.values_list('shot_clock', flat=True).order_by('-game_clock')]

    this_event_index = event.event_index
    previous_scored_event = Event.objects.filter(game__exact=game, event_index__lt=this_event_index, home_score__isnull=False).order_by('-event_index').first()
    if previous_scored_event:
        home_score, visitor_score = previous_scored_event.home_score, previous_scored_event.visitor_score
    else:
        home_score, visitor_score = 0, 0


    # TODO: Is there a better/more Pythonic way to do this?
    # TODO: Couldn't figure out how to do it with pure Django

    speed_data = [11 * [0]]
    for ind in range(len(data)):
        data[ind] = {'marker_data': [{'x': x, 'y': y, 'z': z, 'radius': radius, 'jersey': jersey, 'team_abbrev': team_abbrev,
                                      'h_or_v': 'home' if team_abbrev == game.home.abbreviation else 'visitor',
                                      'name': first_name + ' ' + last_name}
                                      for x, y, z, radius, jersey, team_abbrev, first_name, last_name in
                                      zip(data[ind]['x'], data[ind]['y'], data[ind]['z'], data[ind]['radius'], data[ind]['jersey'], data[ind]['team_abbrev'], data[ind]['first_name'], data[ind]['last_name'])],
                     'game_clock': game_clock_times[ind], 'shot_clock': shot_clock_times[ind]}

        if ind > 0:
            this_ind_speed = [speed_from_coords(cur, prev) for cur, prev in zip(data[ind]['marker_data'], data[ind - 1]['marker_data'])]
            speed_data.append(this_ind_speed)

    big_resp = {'moments_data': data, 'speed_data': speed_data, 'quarter': event.quarter, 'home_score': home_score,
                'visitor_score': visitor_score, 'home_abbrev': game.home.abbreviation, 'visitor_abbrev': game.visitor.abbreviation}

    resp = JsonResponse(big_resp, safe=False)

    return resp


def coach(request):
    if request.method == 'POST':
        form = EventSelector()
    else:
        form = EventSelector()

    return render(request, 'html/coach.html', {'form': form})


def load_events(request):
    # TODO: After I reload the new data with the fix to the parsing script, can get rid of the period_from_pbp__isnull condition
    q_conditions = (Q(has_pbp__exact=True) | Q(period_from_pbp__isnull=False))
    game_ids = request.GET.getlist('games')
    quarters = request.GET.getlist('quarters[]')

    if game_ids:
        q_conditions.add(Q(game_id__in=game_ids), Q.AND)
    if quarters:
        q_conditions.add(Q(quarter__in=quarters), Q.AND)

    events = Event.objects.filter(q_conditions).order_by('event_index')
    return render(request, 'html/dropdown_lists/event_options.html', {'events': events})

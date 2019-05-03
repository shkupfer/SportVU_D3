from django.forms import Form, Select, SelectMultiple, ChoiceField, MultipleChoiceField, CheckboxSelectMultiple
from django.db.models import F, Value, CharField
from django.db.models.functions import Concat
from .models import Game, Event


class EventSelector(Form):
    all_quarters = list(Event.objects.all().values_list('quarter', 'quarter').distinct().order_by('quarter'))

    game = ChoiceField(required=False, choices=[], widget=Select, label="Game: ")
    quarter = MultipleChoiceField(required=False, widget=CheckboxSelectMultiple,
        choices=all_quarters, initial=[v for v, n in all_quarters], label="Quarter: ")
    event = ChoiceField(required=True, choices=[], widget=Select, label="Play: ")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['game'].choices = [("", "Select a game")] + list(Game.objects.all().annotate(game_desc=Concat(F('visitor__name'), Value(' @ '), F('home__name'), Value(', '), F('game_date'), output_field=CharField())).values_list('id', 'game_desc'))
        self.fields['event'].choices = [("", "Select a play")] + list(zip(Event.objects.all().values_list('id', flat=True), Event.objects.all()))

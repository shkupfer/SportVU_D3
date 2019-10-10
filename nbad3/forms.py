from django.forms import Form, Select, SelectMultiple, ChoiceField, MultipleChoiceField, CheckboxSelectMultiple, BooleanField, CheckboxInput
from django.db.models import F, Value, CharField
from django.db.models.functions import Concat
from .models import Game, Event, Possession


class PossessionSelector(Form):
    game = ChoiceField(required=False, choices=[], widget=Select, label="Game: ")
    half_court = BooleanField(required=False, initial=False, label="Only show half court data? ")
    possession = ChoiceField(required=True, choices=[], widget=Select, label="Possession: ")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['game'].choices = [("", "Select a game")] + list(Game.objects.all().annotate(game_desc=Concat(F('visitor__name'), Value(' @ '), F('home__name'), Value(', '), F('game_date'), output_field=CharField())).values_list('id', 'game_desc'))
        # self.fields['possession'].choices = [("", "Select a possession")] + list(zip(Possession.objects.filter(half_court=True).values_list('id', flat=True), Possession.objects.filter(half_court=True)))
        self.fields['possession'].choices = [("", "Select a possession")]
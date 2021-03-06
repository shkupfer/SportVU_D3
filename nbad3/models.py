from django.db.models import Model, ForeignKey, ManyToManyField, CASCADE
from django.db.models import IntegerField, CharField, DateField, DateTimeField, DurationField, FloatField, TimeField, BooleanField


class Team(Model):
    id = IntegerField(primary_key=True)
    name = CharField(max_length=32)
    abbreviation = CharField(max_length=8)
    primary_color = CharField(max_length=32, null=True)
    secondary_color = CharField(max_length=32, null=True)

    def __str__(self):
        return "%s (%s)" % (self.name, self.id)

    class Meta:
        db_table = 'team'


class Game(Model):
    id = IntegerField(primary_key=True)
    game_date = DateField()
    home = ForeignKey(Team, on_delete=CASCADE, related_name='home_rel')
    visitor = ForeignKey(Team, on_delete=CASCADE, related_name='visitor_rel')

    def __str__(self):
        return "%s @ %s, %s" % (self.visitor.name, self.home.name, self.game_date.strftime('%m/%d/%Y'))

    class Meta:
        db_table = 'game'


class Player(Model):
    id = IntegerField(primary_key=True)
    first_name = CharField(max_length=64, null=True)
    last_name = CharField(max_length=64, null=True)

    def __str__(self):
        return "%s %s (%s)" % (self.first_name, self.last_name, self.id)

    class Meta:
        db_table = 'player'


class PlayerStatus(Model):
    player = ForeignKey(Player, on_delete=CASCADE)
    team = ForeignKey(Team, on_delete=CASCADE, null=True)
    # jersey = CharField(max_length=8)
    jersey = CharField(max_length=8, default='', null=True)
    position = CharField(max_length=8)

    def __str__(self):
        return "%s %s, %s. #%s for the %s (%s)" % (self.player.first_name, self.player.last_name, self.position, self.jersey, self.team.name, self.id)

    class Meta:
        db_table = 'player_status'


class Event(Model):
    game = ForeignKey(Game, on_delete=CASCADE)
    eventnum = IntegerField()
    period = IntegerField()
    msg_type = IntegerField()
    msg_action_type = IntegerField()
    ev_real_time = TimeField()
    ev_game_clock = DurationField()
    home_desc = CharField(max_length=256, null=True)
    neutral_desc = CharField(max_length=256, null=True)
    visitor_desc = CharField(max_length=256, null=True)
    home_score_after = IntegerField(default=0)
    visitor_score_after = IntegerField(default=0)
    home_team_fouls_after = IntegerField(default=0)
    visitor_team_fouls_after = IntegerField(default=0)
    person1_type = IntegerField(null=True)
    player1 = ForeignKey(Player, null=True, related_name='p1', on_delete=CASCADE)
    player1_team = ForeignKey(Team, null=True, related_name='p1_team', on_delete=CASCADE)
    person2_type = IntegerField(null=True)
    player2 = ForeignKey(Player, null=True, related_name='p2', on_delete=CASCADE)
    player2_team = ForeignKey(Team, null=True, related_name='p2_team', on_delete=CASCADE)
    person3_type = IntegerField(null=True)
    player3 = ForeignKey(Player, null=True, related_name='p3', on_delete=CASCADE)
    player3_team = ForeignKey(Team, null=True, related_name='p3_team', on_delete=CASCADE)

    def __str__(self):
        return ', '.join([desc for desc in [self.home_desc, self.neutral_desc, self.visitor_desc] if desc is not None]) + ' with %s left in period %s (#%s)' % (self.ev_game_clock, self.period, self.eventnum)

    def desc_for_poss_dropdown(self):
        return ', '.join([desc for desc in [self.home_desc, self.neutral_desc, self.visitor_desc] if desc is not None]) + ' with %s left in period %s' % (self.ev_game_clock, self.period)

    class Meta:
        db_table = 'event'


class Coords(Model):
    moment = ForeignKey('Moment', on_delete=CASCADE, null=True)
    player_status = ForeignKey(PlayerStatus, on_delete=CASCADE)
    x = FloatField()
    y = FloatField()
    z = FloatField()

    class Meta:
        db_table = 'coords'


class Moment(Model):
    game = ForeignKey(Game, on_delete=CASCADE)
    real_timestamp = DateTimeField()
    quarter = IntegerField()
    game_clock = DurationField()
    shot_clock = DurationField(null=True)

    class Meta:
        db_table = 'moment'


class Possession(Model):
    game = ForeignKey(Game, on_delete=CASCADE)
    team = ForeignKey(Team, on_delete=CASCADE)
    start_event = ForeignKey(Event, related_name='start', on_delete=CASCADE)
    end_event = ForeignKey(Event, related_name='end', on_delete=CASCADE)
    points = IntegerField()
    going_to_right = BooleanField(null=True)
    valid = BooleanField(null=True)
    half_court = BooleanField(null=True)
    hc_start = DurationField(null=True)

    def __str__(self):
        return "Possession for team %s from game %s, scored %s points" % (self.team, self.game, self.points)

    def long_desc(self):
        return "Possession for team %s from game %s, scored %s points; Start event: %s; End event: %s" % (self.team, self.game, self.points, self.start_event, self.end_event)

    def dropdown_desc(self):
        return "%s score %s points; start: %s; end: %s" % (self.team.name, self.points, self.start_event.desc_for_poss_dropdown(), self.end_event.desc_for_poss_dropdown())

    class Meta:
        db_table = 'possession'

from django.db.models import Model, ForeignKey, ManyToManyField, CASCADE
from django.db.models import IntegerField, CharField, DateField, DateTimeField, DurationField, FloatField, BooleanField


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
    jersey = CharField(max_length=8, default='')
    position = CharField(max_length=8)

    def __str__(self):
        return "%s %s, %s. #%s for the %s (%s)" % (self.player.first_name, self.player.last_name, self.position, self.jersey, self.team.name, self.id)

    class Meta:
        db_table = 'player_status'


class Event(Model):
    game = ForeignKey(Game, on_delete=CASCADE)
    event_index = IntegerField()
    quarter = IntegerField()
    has_pbp = BooleanField(default=False)
    msg_type = IntegerField(null=True)
    msg_action_type = IntegerField(null=True)
    period_from_pbp = IntegerField(null=True)
    wc_timestring = CharField(max_length=32, null=True)
    pc_timestring = CharField(max_length=32, null=True)
    home_desc = CharField(max_length=256, null=True)
    neutral_desc = CharField(max_length=256, null=True)
    visitor_desc = CharField(max_length=256, null=True)
    home_score = IntegerField(null=True)
    visitor_score = IntegerField(null=True)
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
        return ', '.join([desc for desc in [self.home_desc, self.neutral_desc, self.visitor_desc] if desc is not None]) + ' (#%s)' % self.event_index
        # return "Event %s in quarter %s of game %s (%s)" % (self.event_index, self.quarter, self.game.id, self.id)

    class Meta:
        db_table = 'event'


class Coords(Model):
    player_status = ForeignKey(PlayerStatus, on_delete=CASCADE)
    x = FloatField()
    y = FloatField()
    z = FloatField()

    class Meta:
        db_table = 'coords'


class Moment(Model):
    event = ForeignKey(Event, on_delete=CASCADE)
    real_timestamp = DateTimeField()
    game_clock = DurationField()
    shot_clock = DurationField(null=True)
    coords = ManyToManyField(Coords)

    class Meta:
        db_table = 'moment'

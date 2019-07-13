from nbad3.models import Team, Player, PlayerStatus
import requests
import pandas as pd


ball_team = Team(id=-1, name='_ball', abbreviation='_ball')
ball_player = Player(id=-1, first_name='_ball', last_name='_ball')
# ball_player_status = PlayerStatus(player=ball_player, team=ball_team, jersey=-1, position='')
ball_player_status = PlayerStatus(player=ball_player, team=ball_team, jersey='', position='')


def insert_ball_objs():
    ball_team.save()
    ball_player.save()
    ball_player_status.save()

    return ball_team, ball_player, ball_player_status


team_colors = {'ATL': {'primary': '#E03A3E', 'secondary': '#C1D32F'},
               'BOS': {'primary': '#007A33', 'secondary': '#BA9653'},
               'BKN': {'primary': '#000000', 'secondary': '#FFFFFF'},
               'CHA': {'primary': '#1D1160', 'secondary': '#00788C'},
               'CHI': {'primary': '#CE1141', 'secondary': '#000000'},
               'CLE': {'primary': '#6F263D', 'secondary': '#FFB81C'},
               'DAL': {'primary': '#00538C', 'secondary': '#B8C4CA'},
               'DEN': {'primary': '#0E2240', 'secondary': '#FEC524'},
               'DET': {'primary': '#C8102E', 'secondary': '#006BB6'},
               'GSW': {'primary': '#006BB6', 'secondary': '#FDB927'},
               'HOU': {'primary': '#CE1141', 'secondary': '#000000'},
               'IND': {'primary': '#002D62', 'secondary': '#FDBB30'},
               'LAC': {'primary': '#C8102E', 'secondary': '#1D428A'},
               'LAL': {'primary': '#552583', 'secondary': '#FDB927'},
               'MEM': {'primary': '#5D76A9', 'secondary': '#12173F'},
               'MIA': {'primary': '#98002E', 'secondary': '#000000'},
               'MIL': {'primary': '#00471B', 'secondary': '#EEE1C6'},
               'MIN': {'primary': '#0C2340', 'secondary': '#236192'},
               'NOP': {'primary': '#0C2340', 'secondary': '#85714D'},
               'NYK': {'primary': '#006BB6', 'secondary': '#F58426'},
               'OKC': {'primary': '#007AC1', 'secondary': '#EF3B24'},
               'ORL': {'primary': '#0077C0', 'secondary': '#C4CED4'},
               'PHI': {'primary': '#006BB6', 'secondary': '#ED174C'},
               'PHX': {'primary': '#1D1160', 'secondary': '#E56020'},
               'POR': {'primary': '#E03A3E', 'secondary': '#000000'},
               'SAC': {'primary': '#5A2D81', 'secondary': '#63727A'},
               'SAS': {'primary': '#C4CED4', 'secondary': '#000000'},
               'TOR': {'primary': '#CE1141', 'secondary': '#000000'},
               'UTA': {'primary': '#002B5C', 'secondary': '#00471B'},
               'WAS': {'primary': '#002B5C', 'secondary': '#E31837'},
               '_ball': {'primary': '#EB9423', 'secondary': '#EB9423'}
               }


base_url = "http://stats.nba.com/stats/playbyplayv2?EndPeriod=12&EndRange=55800&GameID=%s&RangeType=2&StartPeriod=1&StartRange=0"
headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
           'Accept-Encoding': 'gzip, deflate',
           'Accept-Language': 'en-US,en;q=0.5',
           'DNT': '1',
           'Upgrade-Insecure-Requests': '1',
           'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'}


def get_pbp_resultset(game_id_int):
    pbp_url = base_url % ("00" + str(game_id_int))
    resp = requests.get(url=pbp_url, headers=headers)
    json_data = resp.json()

    actual_data = json_data['resultSets'][0]
    colnames = actual_data['headers']

    df = pd.DataFrame(actual_data['rowSet'])
    df.columns = colnames

    return df


pbp_keys_translate = {'eventnum': 'EVENTNUM',
                      'msg_type': 'EVENTMSGTYPE',
                      'msg_action_type': 'EVENTMSGACTIONTYPE',
                      'period': 'PERIOD',
                      'home_desc': 'HOMEDESCRIPTION',
                      'neutral_desc': 'NEUTRALDESCRIPTION',
                      'visitor_desc': 'VISITORDESCRIPTION',
                      'person1_type': 'PERSON1TYPE',
                      'player1_id': 'PLAYER1_ID',
                      'player1_team_id': 'PLAYER1_TEAM_ID',
                      'person2_type': 'PERSON2TYPE',
                      'player2_id': 'PLAYER2_ID',
                      'player2_team_id': 'PLAYER2_TEAM_ID',
                      'person3_type': 'PERSON3TYPE',
                      'player3_id': 'PLAYER3_ID',
                      'player3_team_id': 'PLAYER3_TEAM_ID',
                      }

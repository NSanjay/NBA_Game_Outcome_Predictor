from nba_py import Scoreboard
from nba_py import team
from nba_py import game
from nba_py.constants import TEAMS
from datetime import datetime
import pandas as pd
import numpy as np
import datetime
import time

teamToIndex = {
    'ATL': 0,
    'BOS': 1,
    'BKN': 2,  
    'CHA': 3,
    'CHI': 4,
    'CLE': 5,
    'DAL': 6,
    'DEN': 7,
    'DET': 8,
    'GSW': 9,
    'HOU': 10,
    'IND': 11,
    'LAC': 12,
    'LAL': 13,
    'MEM': 14,
    'MIA': 15,
    'MIL': 16,
    'MIN': 17,
    'NOP': 18,
    'NYK': 19,
    'OKC': 20,
    'ORL': 21,
    'PHI': 22,
    'PHX': 23,
    'POR': 24,
    'SAC': 25,
    'SAS': 26,
    'TOR': 27,
    'UTA': 28,
    'WAS': 29,
}

IdToConference = {'1610612737':'Eastern','1610612738':'Eastern','1610612751':'Eastern','1610612766':'Eastern','1610612741':'Eastern','1610612739':'Eastern',
                  '1610612742': 'Western','1610612743':'Western','1610612765':'Eastern','1610612744':'Western','1610612745':'Western','1610612754':'Eastern',
                  '1610612746': 'Western','1610612747':'Western','1610612763':'Western','1610612748':'Eastern','1610612749':'Eastern','1610612750':'Western',
                  '1610612740': 'Western','1610612752':'Eastern','1610612760':'Western','1610612753':'Eastern','1610612755':'Eastern','1610612756':'Western',
                  '1610612757': 'Western','1610612758':'Western','1610612759':'Western','1610612761':'Eastern','1610612762':'Western','1610612764':'Eastern',}

def load_custom_data(teamName,season, opponentTeam):
    #df = team.TeamOpponentSplits(TEAMS[teamName]['id'],location='Home',season=get_season(season),
    #                               opponent_team_id =TEAMS[opponentTeam]['id']).by_opponent()
    df = game.BoxscoreSummary(game_id='0041400122').line_score()
    #df = team.TeamGameLogs(TEAMS[teamName]['id'], get_season(season)).info()
    lf = Scoreboard().east_conf_standings_by_day()
    #df = team.TeamYearOverYearSplits(TEAMS[teamName]['id'], measure_type='Advanced',season=get_season(season), opponent_team_id =TEAMS[opponentTeam]['id']).by_year()
    #df = team.TeamOpponentSplits(TEAMS[teamName]['id'], measure_type='Advanced', season=get_season(season),                                 opponent_team_id=TEAMS[opponentTeam]['id']).by_opponent()
    print(list(lf))
    print(lf)
    print(list(df))
    print(df)
    #print(list(df))
    #return df
    df1 = team.TeamGameLogs(TEAMS[teamName]['id'], get_season(season)).info()
    #df1 = team.TeamLineups(TEAMS['MIL']['id']).overall()
    print(list(df1))
    return df1
    # for row in df.iterrows():
    #     game_id = row["Game_ID"]
    #     game_summary = game.BoxscoreSummary(game_id=game_id).last_meeting()


def load_customized_data(teamName, startYear, endYear):
    home_team = []
    away_team = []
    home_team_home_record_pct = []
    away_team_away_record_pct = []
    home_team_current_win_percentage = []
    away_team_current_win_percentage = []
    home_team_current_standing = []
    away_team_current_standing = []
    home_team_win_percentage_streak_over_last_n_games = []
    away_team_win_percentage_streak_over_last_n_games = []
    home_team_current_streak = []
    away_team_current_streak = []
    recent_head_to_head_wrt_home_team = []

    df = get_teamBoxScore(teamName, get_season(startYear))
    time.sleep(0.5)
    for index, row in df.iterrows():
        game_id = row["Game_ID"]
        print("game_id",game_id)
        game_summary = game.BoxscoreSummary(game_id=game_id).game_summary()
        time.sleep(0.5)
        game_summary = game_summary.iloc[0]

        home_team_id = game_summary["HOME_TEAM_ID"]

        away_team_id = game_summary["VISITOR_TEAM_ID"]
        home_team.append(home_team_id)
        away_team.append(away_team_id)
        date = datetime.datetime.strptime(row['GAME_DATE'],"%b %d, %Y")
        year,month,day = date.year,date.month,date.day
        scoreboard = Scoreboard(month=month,day=day,year=year)
        time.sleep(0.5)
        if IdToConference[str(home_team_id)] == 'Eastern':
            day_home_stats = scoreboard.east_conf_standings_by_day()
        else:
            day_home_stats = scoreboard.west_conf_standings_by_day()

        if IdToConference[str(away_team_id)] == 'Eastern':
            day_away_stats = scoreboard.east_conf_standings_by_day()
        else:
            day_away_stats = scoreboard.west_conf_standings_by_day()

        home_index = np.flatnonzero(day_home_stats['TEAM_ID'] == home_team_id)[0]
        away_index = np.flatnonzero(day_away_stats['TEAM_ID'] == away_team_id)[0]
        day_home_team_stats = day_home_stats.iloc[home_index]
        day_away_team_stats = day_away_stats.iloc[away_index]
        #print("idx::",day_home_team_stats)
        home_team_current_win_percentage.append(day_home_team_stats["W_PCT"])
        away_team_current_win_percentage.append(day_away_team_stats["W_PCT"])
        home_team_current_standing.append(home_index + 1)
        away_team_current_standing.append(away_index + 1)
        #print ("hhghg:",day_home_team_stats["HOME_RECORD"])
        home_wins, home_losses = map(int, day_home_team_stats["HOME_RECORD"].split('-'))
        away_wins, away_losses = map(int, day_away_team_stats["ROAD_RECORD"].split('-'))
        home_team_home_w_pct = 0
        away_team_away_w_pct = 0
        if home_wins + home_losses:
            home_team_home_w_pct = home_wins/(home_wins+home_losses)
        if away_wins + away_losses:
            away_team_away_w_pct = away_wins/(away_wins+away_losses)

        home_team_home_record_pct.append(home_team_home_w_pct)
        away_team_away_record_pct.append(away_team_away_w_pct)


    for i in range(endYear - startYear):
        season = get_season(startYear + 1 + i)
        print("season:::",season)
        additional_data = get_teamBoxScore(teamName, season)
        time.sleep(0.5)
        for index, row in additional_data.iterrows():
            game_id = row["Game_ID"]
            print("game_id::",game_id)
            game_summary = game.BoxscoreSummary(game_id=game_id).game_summary()
            time.sleep(0.5)
            game_summary = game_summary.iloc[0]
            home_team_id = game_summary["HOME_TEAM_ID"]
            away_team_id = game_summary["VISITOR_TEAM_ID"]
            home_team.append(home_team_id)
            away_team.append(away_team_id)
            date = datetime.datetime.strptime(row['GAME_DATE'], "%b %d, %Y")
            year, month, day = date.year, date.month, date.day
            scoreboard = Scoreboard(month=month, day=day, year=year)
            time.sleep(0.5)
            day_stats = None
            if IdToConference[str(home_team_id)] == 'Eastern':
                day_home_stats = scoreboard.east_conf_standings_by_day()
            else:
                day_home_stats = scoreboard.west_conf_standings_by_day()

            if IdToConference[str(away_team_id)] == 'Eastern':
                day_away_stats = scoreboard.east_conf_standings_by_day()
            else:
                day_away_stats = scoreboard.west_conf_standings_by_day()

            try:
                home_index = np.flatnonzero(day_home_stats['TEAM_ID'] == home_team_id)[0]
            except:
                print ("home_team_id::", home_team_id)
                print ("stats::", day_home_stats)
                print ("game_id:::",game_id,game_summary)
                raise Exception("sha")
            away_index = np.flatnonzero(day_away_stats['TEAM_ID'] == away_team_id)[0]
            day_home_team_stats = day_home_stats.iloc[home_index]
            day_away_team_stats = day_home_stats.iloc[away_index]
            home_team_current_win_percentage.append(day_home_team_stats["W_PCT"])
            away_team_current_win_percentage.append(day_away_team_stats["W_PCT"])
            home_team_current_standing.append(home_index + 1)
            away_team_current_standing.append(away_index + 1)
            home_wins, home_losses = map(int,day_home_team_stats["HOME_RECORD"].split('-'))
            away_wins, away_losses = map(int,day_away_team_stats["ROAD_RECORD"].split('-'))
            home_team_home_w_pct = 0
            away_team_away_w_pct = 0
            if home_wins + home_losses:
                home_team_home_w_pct = home_wins / (home_wins + home_losses)
            if away_wins + away_losses:
                away_team_away_w_pct = away_wins / (away_wins + away_losses)

            home_team_home_record_pct.append(home_team_home_w_pct)
            away_team_away_record_pct.append(away_team_away_w_pct)

        df = df.append(additional_data,ignore_index=True)

    home_team_series = pd.Series(home_team)
    away_team_series = pd.Series(away_team)
    home_team_home_record_pct_series = pd.Series(home_team_home_record_pct)
    away_team_away_record_pct_series = pd.Series(away_team_away_record_pct)
    home_team_current_win_percentage_series = pd.Series(home_team_current_win_percentage)
    away_team_current_win_percentage_series = pd.Series(away_team_current_win_percentage)
    home_team_current_standing_series = pd.Series(home_team_current_standing)
    away_team_current_standing_series = pd.Series(away_team_current_standing)

    print("length:::",len(home_team_series.values))
    print("df_length:::",df.index)
    df = df.assign(home_team=home_team_series.values)
    df = df.assign(away_team=away_team_series.values)
    df = df.assign(home_team_home_record_pct=home_team_home_record_pct_series.values)
    df = df.assign(away_team_home_record_pct=away_team_away_record_pct_series.values)
    df = df.assign(home_team_current_win_percentage=home_team_current_win_percentage_series.values)
    df = df.assign(away_team_current_win_percentage=away_team_current_win_percentage_series.values)
    df = df.assign(home_team_current_standing_series=home_team_current_standing_series.values)
    df = df.assign(away_team_current_standing_series=away_team_current_standing_series.values)

    print("headers:::",list(df))
    return df

def load_gameDataWithPlayersAndPointsBetweenYears(teamName, startYear, endYear):
    df = get_gameDataWithPlayersAndPoints(teamName, get_season(startYear))
    for i in range(endYear-startYear):
        season = get_season(startYear+1+i)
        df = df.append(get_gameDataWithPlayersAndPoints(teamName, season), ignore_index=True)
    return df

def get_gameDataWithPlayersAndPoints(teamName, season):
    team_id = TEAMS[teamName]['id']
    df = team.TeamGameLogs(team_id, season).info()
    print(df[:5])
    #   For each game get the players who played and the points and add them to the dataframe line
    for g in df["Game_ID"]:
        print(g)
        player_stats = game.BoxscoreAdvanced(g).sql_players_advanced()
        print("columns:::",list(player_stats))
        print(player_stats)

        players = player_stats["PLAYER_NAME"].values
        minutes = player_stats["MIN"].values
        #points = player_stats["PTS"].values
        teamStartIndices = np.where(player_stats["START_POSITION"].values == 'F')[0]
        home_team_players = players[:teamStartIndices[2]]
        away_team_players = players[:teamStartIndices[2]]

        #   Remove players who didn't play
        notPlayingPlayers = np.where(player_stats["MIN"].isnull())[0]
        players = np.delete(players, notPlayingPlayers)
        #points = np.delete(points, notPlayingPlayers)
        #player_stats["PLAYER_NAME"] = players
        #df = df.assign(player_stats=pd.Series(player_stats).values)
        print("------")
        
        break
    return df

def load_gameDataWithTeamsAndPointsBetweenYears(teamName, startYear, endYear):
    df = get_gameDataWithTeamsAndPoints(teamName, get_season(startYear))
    for i in range(endYear-startYear):
        season = get_season(startYear+1+i)
        print("-----------------")
        print(season)
        df = df.append(get_gameDataWithTeamsAndPoints(teamName, season), ignore_index=True)
        print("-----------------")
    return df

def get_gameDataWithTeamsAndPoints(teamName, season):
    teamsPlayingArray = []
    teamPointsArray = []

    team_id = TEAMS[teamName]['id']
    df = team.TeamGameLogs(team_id, season).info()

    gameIds = df["Game_ID"].values
    #   For each game get the teams who played and the points and add them to the dataframe line
    i = 0
    percentDone = 0
    for matchup in df["MATCHUP"]:
        print( "%s %f " % (teamName, i/len(gameIds)))
        teamsVector = np.zeros(2*len(teamToIndex))
        pointsVector = np.zeros(2*len(teamToIndex))
        team_stats = game.Boxscore(gameIds[i]).team_stats()
        home = 0
        if '@' in matchup:
            home = 0
        else:
            home = 1
        teams = matchup.split(" ")
##        if(teams[0] not in teamToIndex.keys() or )
        t1Index = teamToIndex[teams[0]]
        t2Index = teamToIndex[teams[2]]
        if(team_stats["TEAM_ID"].values[0] == teamName):
            teamNamePoints = team_stats["PTS"].values[0]
            otherPoints = team_stats["PTS"].values[1]
        else:
            teamNamePoints = team_stats["PTS"].values[1]
            otherPoints = team_stats["PTS"].values[0]
        if home:
            teamsVector[t1Index] = 1
            teamsVector[t2Index+len(teamToIndex)] = 1
            
            pointsVector[t1Index] = teamNamePoints
            pointsVector[t2Index+len(teamToIndex)] = otherPoints
        else:
            teamsVector[t2Index] = 1
            teamsVector[t1Index+len(teamToIndex)] = 1
            pointsVector[t2Index] = otherPoints
            pointsVector[t1Index+len(teamToIndex)] = teamNamePoints
        

        teamsPlayingArray.append(teamsVector)
        teamPointsArray.append(pointsVector)
        i+=1
    df = df.assign(teams = pd.Series(teamsPlayingArray).values)
    df = df.assign(points = pd.Series(teamPointsArray).values)
    return df

#   Loads boxScores
def get_teamBoxScore(teamName, season):
    #Use nba_py to load data
    team_id = TEAMS[teamName]['id']
    df = team.TeamGameLogs(team_id, season).info()
    #print("game_logs:::",df)
    return df

def get_season(year):
    CURRENT_SEASON = str(year) + "-" + str(year + 1)[2:]
    return CURRENT_SEASON

#   Loads and adds games for teamName between startYear and endYear seasons to one table
def load_teamBoxScoresBetweenYears(teamName, startYear, endYear):
    df = get_teamBoxScore(teamName, get_season(startYear))
    for i in range(endYear-startYear):
        season = get_season(startYear+1+i)
        df = df.append(get_teamBoxScore(teamName, season), ignore_index=True)
    return df


'''
- These files acquire and prep (remove Nulls, clean up, etc.) all of the provided files for Kaggle's "NFL Big Data Bowl 2023".
- The .csvs have already been downloaded into a folder called 'data' within the repository.
- The functions were built from initial work done in a Jupyter notebook entitled 'data_review.ipynb'.
- The pff scouting reports were used to determine if a blocking player had success blocking (or failed), and vice versa whether a pass rusher had success.
- The pff scouting reports are seperated by role (pass rushers, pass blockers; receivers and quarterbacks not analyzed for now)
'''

import pandas as pd
import numpy as np
import warnings 
warnings.filterwarnings('ignore')


def games():
    '''
    Acquires games and renames columns to be common across all data sources.
    '''
    games = pd.read_csv('data/games.csv')

    games = games.rename(columns = {'gameId':'game'})

    return games


def players():
    '''
    Acquires players and associated data.
    - Note: 'birthDate' and 'collegeName' data has a number of nulls.  An attempt was made to scrape the info for them,
    however it was unsucessful.
    '''
    players = pd.read_csv('data/players.csv')

    # Changes height to integer (inches)
    players['height'] = players.height.str[0].astype(int) * 12 + players.height.str[2:].astype(int)

    # Drop these columns that are full with Nulls and could not be scraped
    players.drop(columns = ['birthDate','collegeName'], inplace = True)

    return players


def plays():
    '''
    Acquires play data and cleans up while adding more relevant features.
    '''
    plays = pd.read_csv('data/plays.csv')

    # Create better yardage metric - yards_to_score - which is the distance to the end zone for the offense 
    plays['yards_to_score'] = np.where(plays.possessionTeam == plays.yardlineSide, 100-plays.yardlineNumber,plays.yardlineNumber)

    # Change clock to seconds remaining in quarter
    plays['gameClock'] = plays.gameClock.str[:2].astype(int) * 60 + plays.gameClock.str[3:].astype(int)

    # Change null dropbacktypes to unknown
    plays['dropBackType'] = np.where(plays.dropBackType.isnull(), 'UNKNOWN',plays.dropBackType)

    # Change home and visitor score to offense and defense score
    game_info = games()
    plays = plays.merge(game_info[['game','homeTeamAbbr','visitorTeamAbbr']], left_on = 'gameId', right_on = 'game', how = 'left')
    plays['o_score'] = np.where(plays.possessionTeam == plays.homeTeamAbbr,plays.preSnapHomeScore,plays.preSnapVisitorScore)
    plays['d_score'] = np.where(plays.defensiveTeam == plays.homeTeamAbbr,plays.preSnapHomeScore,plays.preSnapVisitorScore)

    # Fill penalty/fouls nulls
    plays = plays.fillna(value = {'penaltyYards':0,
                                'foulName1':0,
                                'foulNFLId1':0,
                                'foulName2':0,
                                'foulNFLId2':0,
                                'foulName3':0,
                                'foulNFLId3':0})

    # Drop unecessary columns
    plays = plays.drop(columns = ['yardlineSide',
                                'yardlineNumber',
                                'preSnapHomeScore',
                                'preSnapVisitorScore',
                                'absoluteYardlineNumber',
                                'homeTeamAbbr',
                                'visitorTeamAbbr',
                                'game'])

    # Drop seven rows with Nulls
    plays = plays.dropna()
    
    # Rename and retype columns, as applicable
    plays = plays.rename(columns = {'gameId':'game',
                                    'playId':'play',
                                    'playDescription':'play_description',
                                    'yardsToGo':'yds_togo',
                                    'possessionTeam':'offense',
                                    'defensiveTeam':'defense',
                                    'gameClock':'seconds_left_in_qtr',
                                    'passResult':'pass_result',
                                    'penaltyYards':'penalty_yards',
                                    'prePenaltyPlayResult':'play_outcome',
                                    'playResult':'play_result',
                                    'foulName1':'penalty1',
                                    'foulNFLId1':'penalized1',
                                    'foulName2':'penalty2',
                                    'foulNFLId2':'penalized2',
                                    'foulName3':'penalty3',
                                    'foulNFLId3':'penalized3',
                                    'offenseFormation':'o_formation',
                                    'personnelO':'o_personnel',
                                    'defendersInBox':'d_in_box',
                                    'personnelD':'d_personnel',
                                    'dropBackType':'drop_back',
                                    'pff_playAction':'play_action',
                                    'pff_passCoverage':'coverage',
                                    'pff_passCoverageType':'man/zone'}).astype({'penalty_yards':int,
                                                                                'penalized1':int,
                                                                                'penalized2':int,
                                                                                'penalized3':int,
                                                                                'd_in_box':int})

    return plays


def week(week_num):
    '''
    Creates a dataframe for all games and plays for the week given in the parameters.
    - Note: 'week_num' parameter should just be an integer number, 1-8
    '''
    # Ensure week number is valid
    if week_num not in [1,2,3,4,5,6,7,8]:
        print('Week number not valid.  Week data is for weeks 1 through 8 only.')
        return

    # Format the file path for the desired week
    week = f'data/week{week_num}.csv'

    # Create the base dataframe
    week = pd.read_csv(week)

    # Strip unecessary columns, rename fill NaNs (all for the 'football' and retype)
    week = week.drop(columns = ['time','playDirection','team','jerseyNumber']).rename(columns = {'gameId':'game',
                                                                        'playId':'play',
                                                                        'frameId':'frame',}).fillna(0).astype({'nflId':int})

    return week


def scout_pass_rush():
    '''
    Aquires scout data then isolates players who rush the passer on a given play to determine if they were able to pressure the qb (hit, hury or sack)
    - Note: Does not include those in coverage who then rush the passer.
    '''
    scout = pd.read_csv('data/pffScoutingData.csv')

    # Isolate pass rushers
    scout_pass_rush = scout[scout.pff_role == 'Pass Rush']

    # Strip unnecessary columns, rename and retype
    scout_pass_rush = scout_pass_rush[['gameId',
                                    'playId',
                                    'nflId',
                                    'pff_positionLinedUp',
                                    'pff_hit',
                                    'pff_hurry',
                                    'pff_sack',]].rename(columns = {'gameId':'game',
                                                                    'playId':'play',
                                                                    'pff_positionLinedUp':'position',
                                                                    'pff_hit':'hit',
                                                                    'pff_hurry':'hurry',
                                                                    'pff_sack':'sack'}).astype({'hit':int,
                                                                                                'hurry':int,
                                                                                                'sack':int})

    # Create a catch-all column for pressure
    scout_pass_rush['pressure'] = scout_pass_rush.hit + scout_pass_rush.hurry + scout_pass_rush.sack

    return scout_pass_rush


def scout_pass_block():
    '''
    Aquires scout data then filters players who make or attempt a block at some point during a play, along with if their block fails (beaten, hit, hurry or sack allowed)
    '''
    scout = pd.read_csv('data/pffScoutingData.csv')

    # Isolate pass blockers (includes those with role 'pass block' taht do not engage a defender, as well as those receivers who block at some point in the play)
    scout_pass_block = scout[(scout.pff_role == 'Pass Block') | (scout.pff_nflIdBlockedPlayer.notnull() == True)]

    # Replace one null value
    scout_pass_block.at[71367,'pff_backFieldBlock'] = 0.0

    # Fill in missing blocked player, blocktype and backfield block values
    scout_pass_block = scout_pass_block.fillna(value = {'pff_nflIdBlockedPlayer':0,
                                                    'pff_blockType':'NB',
                                                    'pff_backFieldBlock':0})

    # Strip unnecessary columns, rename and retype
    scout_pass_block = scout_pass_block[['gameId',
                                    'playId',
                                    'nflId',
                                    'pff_positionLinedUp',
                                    'pff_nflIdBlockedPlayer',
                                    'pff_blockType',
                                    'pff_backFieldBlock',
                                    'pff_beatenByDefender',
                                    'pff_hitAllowed',
                                    'pff_hurryAllowed',
                                    'pff_sackAllowed']].rename(columns = {'gameId':'game',
                                                                    'playId':'play',
                                                                    'pff_positionLinedUp':'position',
                                                                    'pff_beatenByDefender':'beaten_by_pass_rusher',
                                                                    'pff_hitAllowed':'hit_allowed',
                                                                    'pff_hurryAllowed':'hurry_allowed',
                                                                    'pff_sackAllowed':'sack_allowed',
                                                                    'pff_nflIdBlockedPlayer':'rusher_blocked',
                                                                    'pff_blockType':'block_type',
                                                                    'pff_backFieldBlock':'backfield_block'}).astype({'beaten_by_pass_rusher':int,
                                                                                                                     'hit_allowed':int,
                                                                                                                     'hurry_allowed':int,
                                                                                                                     'sack_allowed':int,
                                                                                                                     'rusher_blocked':int,
                                                                                                                     'backfield_block':int})

    # Create a catch-all column for pressure allowed (blocking failure)
    scout_pass_block['block_fail'] = scout_pass_block.beaten_by_pass_rusher + scout_pass_block.hit_allowed + scout_pass_block.hurry_allowed + scout_pass_block.sack_allowed

    return scout_pass_block


def get_players_in_play(game, play):
    '''
    Gets all of the NFL player ids along with their role for a given play.    
    '''
    scout_players = pd.read_csv('data/pffScoutingData.csv')

    players = scout_players[['gameId',
                            'playId',
                            'nflId',
                            'pff_role'
                            'pff_positionLinedUp']].rename(columns = {'gameId':'game',
                                                            'playId':'play',
                                                            'pff_role':'role',
                                                            'pff_positionLinedUp':'position'})

    players = players[players.game == game][players.play == play]

    return players


def all_plays():
    '''
    Creates a data object with all plays from the 8 weeks.
    * I have a feeling this is not the most efficient way to do this...
    '''
    # Load the plays dataframe
    plays = plays()

    # Create the data structure to return {game: [{play: [players]}]}
    game_play_players = {}  

    # Loop through all of the games to pull out their plays and the players invovled
    for game in plays.gameId.unique():
        game_play_list = {}

        for play in plays[plays.gameId == game].playId.unique():
            play_player_dict = {}
            play_player_list = []

            for player in plays[plays.playId == play][plays.gameId == game].nflId:
                play_player_list.append(player)

            play_player_dict[play] = play_player_list
            game_play_list.update(play_player_dict)

        game_play_players[game] = game_play_list

    return game_play_players
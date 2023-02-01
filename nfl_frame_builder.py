import numpy as np
import pandas as pd

from scipy import spatial

import warnings
warnings.filterwarnings('ignore')

import nfl_acquire_and_prep as acquire


# ====================================================================================================
# CREATE PLAY FRAMES
# ==================================================================================================== 

def get_play_frames(game, play, week_df):   # def get_play_frames(game, play): # Week pulled out to make things run faster
    '''
    Takes in a game and play and returns the movement of all players plus the ball in .1 second frames 
    over the course of the play.  Necessary to create metrics that rely on quasi-continuous data.

    Parameters:
        'game' - Integer - Game number (unique for season)
        'play' - Integer - Unique for game
        'week_df' - Dataframe - Weekly frame by frame data for each play 
    Returns: 
        'play_frames_df' - Dataframe - All players (nflId) and the ball (nflId = 0) and their movement data.
    '''
    # # Find the proper week to analyze and bring that dataframe in
    # week_num = get_week_of_game(game)

    # # Acquire that week's df
    # week_df = acquire.week(week_num)

    # Extract frames for a given play
    play_frames_df = week_df[week_df.game == game][week_df.play == play]
    
    # Add in the shifts
    play_frames_df = add_next(play_frames_df)

    return play_frames_df


# ----- Support Functions -----------------------------------------------------------------------------

def get_week_of_game(game):
    '''
    Gets the week number of a given game, which is needed to access the proper weekly data csv.

    Parameters:
        'game' - Integer - The unique number given to the game
    Returns: 
        'week_num' - Integer - The week number for a given game
    '''
    # Acquire small df with game + week information
    games = acquire.games()

    # Match the game to a week
    week_num = games.loc[games.game == game, 'week']

    return int(week_num)


def add_next(play_frames_df):
    '''
    Adds in the next location (x and y coordinates) to use for metric building.

    Parameters:
        'play_frames_df' - Dataframe - Play player movement data for a given play
    Returns:
        'play_frames_df' - Dataframe - Play player movement data for a given play
    '''
    # Create new column with x and y locations shifted one frame forward
    play_frames_df['next_x'] = play_frames_df.x.shift(-1)
    play_frames_df['next_y'] = play_frames_df.y.shift(-1)
    
    return play_frames_df




# ====================================================================================================
# BUILD BASE PLAY FRAME
# ====================================================================================================

def build_play_frames(game, play, week_df, nflId, scout_pass_block, player_type, v_type = 'PvB'): # build_play_frames(game, play, nflId, player_type, v_type = 'PvB'): # 
    '''
    Combines the function which creates the play frames with the function that creates the play frames.
    
    Parameters:
        'game' - Integer - Game number (unique for season)
        'play' - Integer - Unique for game
        'week_df' - Dataframe - Weekly frame by frame data for each play 
        'nflId' - Integer - Unique Id of player being analyzed
        'scout_pass_block' - Dataframe - Pass blocker PFF scouting data
        'player_type' - String - The type of player bring analyzed (pass_rusher or pass_blocker)
        'v_type' - String - The type of analysis to perform, player vs ball or player vs player (and ball.)  Either 'PvB' or 'PvP'.
    Returns:
        'qb_hold_time' - Float - How long the qb holds the ball (in seconds)
        'point_of_scrimmage' - Tuple of Floats - x and y coordinates of snap
        'analysis_frames' - Dataframe - Complete focused frames of the play ready for analysis
    '''
    play_frames_df = get_play_frames(game, play, week_df) # play_frames_df = get_play_frames(game, play)
    
    qb_hold_time, point_of_scrimmage, analysis_frames = create_play_analysis_frames(play_frames_df,
                                                                                    nflId,
                                                                                    scout_pass_block, ### 
                                                                                    player_type = player_type,
                                                                                    v_type = v_type)
    
    return qb_hold_time, point_of_scrimmage, analysis_frames


# ----- Sub Functions --------------------------------------------------------------------------------

def create_play_analysis_frames(play_frames_df, nflId, scout_pass_block, player_type, v_type): # def create_play_analysis_frames(play_frames_df, nflId, player_type, v_type) 
    '''
    Create the play frames, consisting of the curent and next x and y coordiantes, as well as the acceleration
    at the time of the frame, for the ball and pass rusher, as well as opponents if PvP is selected.
    PvB is used only to analyze pass rushers without accounting for any blockers.
    
    Parameters:
        'play_frames_df' - Dataframe - All players (nflId) and the ball (nflId = 0) and their movement data.
        'nflId' - Integer - Unique Id of player being analyzed
        'scout_pass_block' - Dataframe - Pass blocker PFF scouting data
        'player_type' - String - The type of player bring analyzed (pass_rusher or pass_blocker)
        'v_type' - String - The type of analysis to perform, player vs ball or player vs player (and ball.)  Either 'PvB' or 'PvP'.
    Returns:
        'qb_hold_time' - Float - How long the qb holds the ball (in seconds)
        'point_of_scrimmage' - Tuple of Floats - x and y coordinates of snap
        'play_fb_frames' - Dataframe - Cleaned frames ready to index players off of 
    '''
    # Get football frames: used for all
    qb_hold_time, point_of_scrimmage, play_fb_frames = get_play_fb_frames(play_frames_df)
    
    # Get game and play
    game = int(play_frames_df.game.unique())
    play = int(play_frames_df.play.unique())

    # Use the matchup function to see what kind of player the NFL ID is, and who they went against in the scouting data
    # player_type, opponent_type, opponents = matchup_finder(game, play, nflId, player_type)
    player_type, opponent_type, opponents = matchup_finder(game, play, nflId, player_type, scout_pass_block)
    
    # Get the number of opponents (used for labeling)
    opponent_count = len(opponents)
    
    if v_type == 'PvB':
        # Get player
        player_frames_df = get_play_player_frames(play_frames_df, nflId)
        
        # Rename player analyze to pass_rusher or pass_blocker
        player_frames_df = rename_player(player_frames_df, player_type)
        
        # Create a dataframe which pairs the location and movement with the football and the player frame by frame (0.1 second intervals)
        pvb_analysis_frames = merge_frames(play_fb_frames, player_frames_df)
        
        # Drop first and last frames - while losing a small amount of data improves overall analysis
        pvb_analysis_frames = pvb_analysis_frames[1:-1]
        
        # Reset frames
        pvb_analysis_frames = pvb_analysis_frames.reset_index(drop = True)

        return qb_hold_time, point_of_scrimmage, pvb_analysis_frames        
    
    elif v_type == 'PvP':
        # Get player
        player1_frames_df = get_play_player_frames(play_frames_df, nflId)
        
        # Rename player analyzed
        player1_frames_df = rename_player(player1_frames_df, player_type)
        
        # Create a dataframe which pairs the location and movement with the football and the player frame by frame (0.1 second intervals)
        pvp_analysis_frames = merge_frames(play_fb_frames, player1_frames_df)
    
        if opponent_count == 1:
            
            player = opponents[0]
        
            opponent_frames = get_play_player_frames(play_frames_df, player)
            
            pvp_analysis_frames = merge_frames(pvp_analysis_frames, opponent_frames)
            
            # Rename opponent
            pvp_analysis_frames = rename_opponents(pvp_analysis_frames, opponent_count, opponent_type)
    
        elif opponent_count > 1:
            
            # Create a counter for the rename function to keep track of
            count = 0
            
            # Now use this list to add an integer lable to each opponent's columns
            for player in opponents:
                # Up the counter for the rename function
                count += 1
                opponent_frames = get_play_player_frames(play_frames_df, player)
                
                opponent_frames = rename_opponents(opponent_frames, opponent_count, opponent_type, count = count)

                pvp_analysis_frames = merge_frames(pvp_analysis_frames,opponent_frames)     
        
        else:
            print('Player did not have any opponents listed, cannot do PvP, showing PvB instead')            
        
        # Drop first and last frames - while losing a small amount of data improves overall analysis
        pvp_analysis_frames = pvp_analysis_frames[1:-1] 
        
        # Reset frames
        pvp_analysis_frames = pvp_analysis_frames.reset_index(drop = True)
        
        return qb_hold_time, point_of_scrimmage, pvp_analysis_frames
        
    else:
        print('v_type inputs are either:\n-"PvB" to compare pass rusher to ball/QB; or:\n-"PvP" to compare pass rusher to blocker')


def get_play_fb_frames(play_frames_df):
    '''
    Isolates the football frames (football movement over the course of the play) for a given play.
    
    Parameters:
        'play_frames_df' - Dataframe - All players (nflId) and the ball (nflId = 0) and their movement data.
    Returns:
        'qb_hold_time' - Float - How long the qb holds the ball (in seconds)
        'point_of_scrimmage' - Tuple of Floats - x and y coordinates of snap
        'play_fb_frames' - Dataframe - Cleaned frames ready to index players off of     
    '''
    # nflId for the football is 0, creates a dataframe of the football over the play
    play_fb_frames = play_frames_df[play_frames_df.nflId == 0].set_index('frame',drop = True)

    # Get the x value for the line of scrimmage (for use with play graphing)
    point_of_scrimmage = (play_fb_frames.x.iloc[0], play_fb_frames.y.iloc[0])
    
    # Clean the frames
    play_fb_frames = clean_fb_frames(play_fb_frames)
    
    # Drop the last frame since the shifted columns will have Nulls
    play_fb_frames = play_fb_frames[:-1]
    
    # Get the start and end frames of the qb with the ball - allows for qb hold time even if truncate = False
    snap_frame, end_frame = determine_pertinent_frames(play_fb_frames)
    
    # Calculate qb hold time
    qb_hold_time = (end_frame - snap_frame)/10
    
    # Remove all frames before snap and after event where ball is no longer being pass rushed
    play_fb_frames = play_fb_frames[(play_fb_frames.index >= snap_frame) & (play_fb_frames.index <= end_frame)]

    # Drops event, as not longer necessary
    play_fb_frames = play_fb_frames.drop(columns = ['event'])

    return qb_hold_time, point_of_scrimmage, play_fb_frames


def get_play_player_frames(play_frames_df, nflId):
    '''
    Creates a dataframe which isolates the player's frame by frame movement data.
    
    Parameters:
        'play_frames_df' - Dataframe - All players (nflId) and the ball (nflId = 0) and their movement data.
        'nflId' - Integer - Unique Id of player being analyzed
    Returns:
        'player_frames_df' - Dataframe - Frames of selected player
    '''
    # Merge will truncate this if needed later
    player_frames_df = play_frames_df[play_frames_df.nflId == nflId].set_index('frame',drop = True)
    
    # Clean the frames
    player_frames_df = clean_player_frames(player_frames_df)
    
    # Drop the last row, as it is corrupted by the player before it
    player_frames_df = player_frames_df[:-1]

    return player_frames_df


# ----- Support Functions -----------------------------------------------------------------------------

def clean_fb_frames(play_fb_frames):
    '''
    Cleans up the football frames (nflId = 0) of the play movement dataframe.

    Parameters:
        'play_fb_frames' - Dataframe - Football movement over play, 'dirty'
    Returns:
        'play_fb_frames' - Dataframe - Football movement over play, now cleaned
    '''
    # Drop unecessary columns
    play_fb_frames = play_fb_frames.drop(columns = ['game',
                                                    'play',
                                                    'nflId',
                                                    's',
                                                    'o',
                                                    'a',
                                                    'dis',
                                                    'dir']).rename(columns = {'x':'ball_x',
                                                                              'y':'ball_y',
                                                                              'next_x':'ball_next_x',
                                                                              'next_y':'ball_next_y'})
    
    # Re-order columns
    play_fb_frames = play_fb_frames[['event',
                                     'ball_x',
                                     'ball_y',
                                     'ball_next_x',
                                     'ball_next_y']]
    
    return play_fb_frames


def clean_player_frames(play_player_frames):
    '''
    Cleans up the player frames (nflId = 0) of the play movement dataframe.

    Parameters:
        'play_player_frames' - Dataframe - Player movement over play, 'dirty'
    Returns:
        'play_player_frames' - Dataframe - Player movement over play, now cleaned
    '''
    # Drop unecessary columns, and rename remaining
    play_player_frames = play_player_frames.drop(columns = ['game',
                                                            'play',
                                                            's',
                                                            'o',
                                                            'dir',
                                                            'dis',
                                                            'event']).rename(columns = {'x':'player_x',
                                                                                        'y':'player_y',                                                                    
                                                                                        'a':'player_a',
                                                                                        'next_x':'player_next_x',
                                                                                        'next_y':'player_next_y'})
    # Re-order columns
    play_player_frames = play_player_frames[['nflId',
                                             'player_x',
                                             'player_y',
                                             'player_next_x',
                                             'player_next_y',
                                             'player_a']]
    
    return play_player_frames


def determine_pertinent_frames(play_fb_frames):
    '''
    This function determines the indices of the relevant frames for pass rush analysis.
    It parses the frame events for a starting (snap) index and the ending index for the analysis.
    
    Parameters:
        'play_fb_frames' - Dataframe - Dataframe of fb movement (post-cleaning)
    Returns:
        'snap_index' - Integer - Starting point for truncated frames (the 'snap' event)
        'end_index' - Integer - End point for truncated frames (after ball leave qb event: see list in function)
    '''
    # Initiate a trigger which tells the function to start checking the events after ball snap for the end event (when qb passes, is sacked, etc.)
    trigger = 0
    
    for i, event in enumerate(play_fb_frames.event):
        # If the event is a ball_snap, stores the snap index and triggers the function to start checking events
        if event == 'ball_snap':
            snap_index = i + 1
            trigger = 1
            continue
        
        if trigger == 1:
            # The following events are not end events
            if event in ['None','autoevent_ballsnap','autoevent_passforward','play_action','first_contact','shift','man_in_motion','line_set']:
                continue
            # If the trigger is on and the event is an end event, return the index
            else:
                return snap_index, i + 1 #-> i + 1 being the end index
        else:
            continue


def matchup_finder(game, play, nflId, player_type, scout_pass_block): # matchup_finder(game, play, nflId, player_type):
    '''
    Used in PvP find the pass blocker(s) blocking the pass rusher, or the pass rusher opposing the blockers
    
    Parameters:
        'game' - Integer - Game number (unique for season)
        'play' - Integer - Unique for game
        'nflId' - Integer - The unique id for the primary player being analyzed
        'player_type' - String - The type of player bring analyzed (pass_rusher or pass_blocker)
        'scout_pass_block' - Dataframe - Pass blocker PFF scouting data
    Returns: 
        'player_type' - String - The type of player bring analyzed (pass_rusher or pass_blocker)
        'opponent_type' - String - The type of player opposing the analyzed player
        'opponent_list' - List of ints - List of nflId ints (players) pass rushing/blocking pass rusher
    '''
    # scout_pass_block = acquire.scout_pass_block()

    scout_pass_block_play = scout_pass_block[scout_pass_block.game == game][scout_pass_block.play == play]
    
    # If player's nflId is in this list, they are a pass blocker and opponent is pass rusher
    if nflId in list(scout_pass_block_play.nflId):
        # Input check of player_type
        if player_type == 'pass_rusher':
            print('Error: Player types do not match.  Check your the player type you inputted as a parameter')
            return

        opponent_type = 'pass_rusher'
        
        return player_type, opponent_type, list(scout_pass_block_play[scout_pass_block_play.nflId == nflId].rusher_blocked)
    
    elif nflId in list(scout_pass_block_play.rusher_blocked):
        if player_type == 'pass_blocker':
            print('Error: Player types do not match.  Check your the player type you inputted as a parameter')
            return 
            
        opponent_type = 'pass_blocker'
        
        return player_type, opponent_type, list(scout_pass_block_play[scout_pass_block_play.rusher_blocked == nflId].nflId)
    
    # In order to not error out in later function if there are no matchups, need to return a blank string and blank list
    else:
        return player_type, '', []


def merge_frames(df1, df2):
    '''
    Puts together different frames generated from the weekly data.
    Note: Ensure you always use the football frames as the leftmost df1 if you have multiple players to merge.

    Parameters:
        'df1' - Dataframe - Left frame to merge on
        'df2' - Dataframe - Right frame to merge on df1
    Returns:
        'merged_frames' - Dataframe - df1 and df2 merged on 'frame' column
    '''
    merged_frames = df1.merge(df2, on = 'frame', how = 'left')
    
    return merged_frames


def rename_player(player_frames_df, player_type):
    '''
    Renames the main player being analyzed by typing them (pass rusher or blocker)
    
    Parameters:
        'player_frames_df' - Dataframe - Frames of selected player
        'player_type' - String - The type of player bring analyzed (pass_rusher or pass_blocker)
    Returns:
        'player_frames_df' - Dataframe - Frames of selected player (now renamed to player type)

    '''
    player_frames_df = player_frames_df.rename(columns = {'nflId':f'{player_type}',
                                                          'player_a':f'{player_type}_a',
                                                          'player_x':f'{player_type}_x',
                                                          'player_y':f'{player_type}_y',
                                                          'player_next_x':f'{player_type}_next_x',
                                                          'player_next_y':f'{player_type}_next_y'})
    
    return player_frames_df


def rename_opponents(opponent_frames, opponent_count, opponent_type, count = 0):
    '''
    Renames opposing players to make understanding and manipulations easier.
    
    Parameters:
        'opponent_frames' - Dataframe - Play frames of the opponent
        'opponent_count' - Integer - How many opponents will be analyzed
        'opponent_type' - String - Like player_type, is 'pass_rusher' or 'pass_blocker'
        'count' - Integer - Used for naming opponents
    Returns:
        'opponent_frames' - Dataframe - Play frames of the opponent (now with new names)
    '''
    # If there is a single opponent
    if opponent_count == 1:
        #Change the column names to nflId and opponent to make easier
        opponent_frames = opponent_frames.rename(columns = {'nflId':f'{opponent_type}',
                                                            'player_x':f'{opponent_type}_x',
                                                            'player_y':f'{opponent_type}_y',
                                                            'player_a':f'{opponent_type}_a',
                                                            'player_next_x':f'{opponent_type}_next_x',
                                                            'player_next_y':f'{opponent_type}_next_y'})

    else:
        opponent_frames = opponent_frames.rename(columns = {'nflId':f'{opponent_type}_{count}',
                                                            'player_x':f'{opponent_type}_{count}_x',
                                                            'player_y':f'{opponent_type}_{count}_y',
                                                            'player_a':f'{opponent_type}_{count}_a',
                                                            'player_next_x':f'{opponent_type}_{count}_next_x',
                                                            'player_next_y':f'{opponent_type}_{count}_next_y'})
        
    return opponent_frames




# ====================================================================================================
# TESTING FUNCTIONS
# ====================================================================================================

def random_play():
    '''
    Creates a random play to analyze from the entire 8-week season.
    Play numbers can be assigned in different weeks, so they are not unique, hence the need to specify game.

    Parameters:
        NONE
    Returns:
        'game' - Integer - Game number (unique for season)
        'play' - Integer - Unique for game
    '''
    # Pull in data from acquire function that creates an object full of game-play combinations
    game_play_players = acquire.all_plays()

    # Randomly choose a game then a play from that game
    game = np.random.choice(list(game_play_players.keys()))
    play = np.random.choice(list(game_play_players[game]))

    return game, play


# ----- Support Functions -----------------------------------------------------------------------------

def get_players_in_play(game, play):
    '''
    Gets all of the NFL player ids along with their role and position for a given play.  
    
    Parameters:
        'game' - Integer - Game number (unique for season)
        'play' - Integer - Unique for game
    Returns:
        'play_players' - Dataframe - A dataframe of play players and their roles and positions
    '''
    # Load data
    scout_players = pd.read_csv('data/pffScoutingData.csv')

    # Clean and rename
    play_players = scout_players[['gameId',
                                  'playId',
                                  'nflId',
                                  'pff_role',
                                  'pff_positionLinedUp']].rename(columns = {'gameId':'game',
                                                                            'playId':'play',
                                                                            'pff_role':'role',
                                                                            'pff_positionLinedUp':'position'})

    # Isolate players from given game and play
    play_players = play_players[play_players.game == game][play_players.play == play].drop(columns = ['game',
                                                                                                      'play'])

    return play_players
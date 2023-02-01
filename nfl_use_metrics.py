'''


'''


import nfl_frame_builder as nfl
import nfl_acquire_and_prep as acquire
import nfl_build_metrics as metrics

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')


# ====================================================================================================
# BUILD PLAYER AND PLAY METRICS DATAFRAME, BY WEEK
# ====================================================================================================

def all_week_pass_rush_results(start_week = 1, end_week = 8):
    '''
    Self-contained function that pulls in all the metrics for desired weeks and outputs pass_rush_results.
    
    Note: Week range for Data Bowl is 1-8.  Can use 1-6 for training data for any ML analysis performed.
    
    Parameters:
        'start_week' - Integer - Week to start building from (inclusive)
        'end_week' - Integer - Week to build to (inclusive)
    Returns:
        'all_results' - Dataframe - Metrics for each player-play for given weeks
        ***.csv of results saved to folder***
    '''
    # Load the data used within the sub-functions
    players_df = acquire.players()
    scout_pass_rush = acquire.scout_pass_rush()
    scout_pass_block = acquire.scout_pass_block()
    # Default to 'PvP'
    v_type = 'PvP'
    
    # Create empty dataframe to hold results
    all_results = pd.DataFrame()
    
    # Run through weeks to build for    
    for i in range(start_week, end_week + 1):
        print('2021 NFL Week:',i)
        # Acquire that week's frame data
        week_df = acquire.week(i)
    
        pass_rush_results = play_player_metrics_builder(scout_pass_rush, scout_pass_block, v_type, players_df, week_df)
        
        all_results = pd.concat([all_results, pass_rush_results])
        
    # Save to a csv for easier later use
    filename = f'metric_results_weeks_{start_week}_through_{end_week}.csv'
    all_results.to_csv(filename, index = False)
        
    return all_results


# ----- Sub Functions -----------------------------------------------------------------------------

def pull_metrics(analysis_frames, qb_hold_time):
    '''
    Consolidates metrics for a given play.
    
    Parameters:
        'analysis_frames' - Dataframe - Complete frames of the play.
        'qb_hold_time' - Float - How long the qb holds the ball (in seconds)
    Returns:
        'play_metrics' - Dictionary - Contains aggregated play metrics (averages over frames)
    '''
    # Pass Rusher Stats from analysis_frame
    pass_rusher = analysis_frames.pass_rusher.max()
    pass_rusher_average_a = round(analysis_frames.pass_rusher_a.mean(),4)
    colinearity = round(analysis_frames.colinearity.mean(),4)
    pursuit_factor = round(analysis_frames.pursuit_factor.mean(),4)
    force_to_ball = round(analysis_frames.pass_rusher_force_to_ball.mean(),4)
    #! -- Add any additional frame by frame metrics below --
    pursuit_vs_escape = round(analysis_frames.pursuit_vs_escape.mean(),4)
    pursuit1 = round(analysis_frames.pursuit1.mean(),4)
    pursuit2 = round(analysis_frames.pursuit2.mean(),4)
    pursuit3_mean = round(analysis_frames.pursuit3.mean(),4)
    pursuit3_sum = round(analysis_frames.pursuit2.sum(),4)
    pursuit4 = round(analysis_frames.pursuit4.mean(),4)

    
    # List pass blockers in play - not needed in PvB
    pass_blockers = metrics.get_pass_blockers(analysis_frames)
    pass_blocker_nflId_list = []
    for pass_blocker in pass_blockers:
        pass_blocker_nflId_list.append(analysis_frames[pass_blocker].max())
        
    # Create dictionary object to hold results
    play_metrics = {'pass_rusher':pass_rusher,
                    'pass_rusher_average_a':pass_rusher_average_a,
                    'colinearity':colinearity,
                    'pursuit_factor':pursuit_factor,
                    'force_to_ball':force_to_ball,
                    #! -- Add any additional frame by frame metrics --
                    'pursuit_vs_escape':pursuit_vs_escape,
                    'pursuit1':pursuit1,
                    'pursuit2':pursuit2,
                    'pursuit3_mean':pursuit3_mean,
                    'pursuit3_sum':pursuit3_sum,
                    'pursuit4':pursuit4,
                    'qb_hold_time':qb_hold_time,
                    'blocker_count':len(pass_blocker_nflId_list), # Not needed in PvB
                    'pass_blockers':pass_blocker_nflId_list} # Not needed in PvB
        
    return play_metrics


def play_player_metrics_builder(scout_pass_rush, scout_pass_block, v_type, players_df, week_df):
    '''
    Given a specific week and its associated dataframe, create the metrics for each player in each play and then merge
    with player-play results from PFF's scouting reports.
    
    Note: For final analysis only need to do this for player_type = 'pass_rusher' as we can then pull the 'pass_blocker'
    metrics from these.
    
    Parameters:
        'scout_pass_rush' - Dataframe - Pass rusher PFF scouting data
        'scout_pass_block' - Dataframe - Pass blocker PFF scouting data
        'v_type' - String - The type of analysis to perform, defaulting to 'PvP'
        'players_df' - Dataframe - Contains player information, including weight
        'week_df' - Dataframe - Weekly frame by frame data for each play 
    Returns:
        'pass_rush_results' - Dataframe - Metrics for each player in each play, with pressure statistics from PFF scouting
    '''
    results = pd.DataFrame()
    
    week_game_list = week_df.game.unique()
    
    scout_pass_rush = scout_pass_rush[scout_pass_rush.game.isin(week_game_list)]
    
    for entry in scout_pass_rush.index:
        # Added a try except since there are errors when the snap events are missing
        try:
            game = scout_pass_rush.game.loc[entry]
            play = scout_pass_rush.play.loc[entry]
            nflId = scout_pass_rush.nflId.loc[entry]

            qb_hold_time, point_of_scrimmage, analysis_frames = nfl.build_play_frames(game,
                                                                                      play,
                                                                                      week_df,
                                                                                      nflId,
                                                                                      scout_pass_block,
                                                                                      player_type = 'pass_rusher',
                                                                                      v_type = v_type)

            analysis_frames = metrics.build_metrics(analysis_frames,
                                                    point_of_scrimmage,
                                                    players_df)

            play_metrics = pd.DataFrame([pull_metrics(analysis_frames, qb_hold_time)])
            
            play_metrics = pd.merge(scout_pass_rush[scout_pass_rush.index == entry],
                                    play_metrics,
                                    how = 'inner',
                                    left_on = 'nflId',
                                    right_on = 'pass_rusher')

            results = pd.concat([results, play_metrics])

            print(f'Added game|play|nflId = {game}|{play}|{nflId}')
            
        except:
            print(f'*****Frame event error for game|play|nflId = {game}|{play}|{nflId}')
    
    pass_rush_results = results.drop(columns = ['pass_rusher'])
    
    return pass_rush_results
import numpy as np
import pandas as pd

from scipy import spatial

import nfl_acquire_and_prep as acquire

'''
The following functions are built to create novel features and metrics for various levels of analysis (play, game, season.
They can also produce simple graphical representations of plays to compare with these metrics and features.
The function 'full_analysis' emits a dataframe where each pass rusher has a number of metrics along with basic pass
rush stats.

ToDo:
1. Add option to pick metrics, beyond 'return pursuit angle'
    - Perhaps create a seperate .py file full of functions to use
2. Rename this file as defensive_nfl_functions, and create new .py file that can pull from this to analyze ofenseive players
vs. defensive players of this metric.
3. Finish commenting and all that jazz
'''

# -----------------------------------------------------------------------------------------------------------------
# METRIC ANALYSIS FUNCTIONS
# -----------------------------------------------------------------------------------------------------------------

def full_analysis(return_pursuit_angle = True):
    '''
    Emits a dataframe with each pass rusher's metrics for weeks 1-8, along with the outcome from the scouting report.
    '''
    # Acquire pass rushers
    pass_rushers_df = acquire.scout_pass_rush()
    
    # Create dataframe to hold results
    all_game_metrics = pd.DataFrame()
    
    # Iterate through each weekly dataset and run function to acquire that week's metrics
    for i in range(8):
        week_df = acquire.week(i+1)
        
        week_metrics = week_analysis(week_df, pass_rushers_df, return_pursuit_angle = return_pursuit_angle)
        
        all_game_metrics = pd.concat([all_game_metrics, week_metrics])
        
    return all_game_metrics


def week_analysis(week_df, pass_rushers_df, return_pursuit_angle = True):
    '''
    Emits a dataframe with each pass rushers metrics for the week, along with the outcome from the scouting report.
    '''
    # Get a dict of game:plays for the chosen week
    week_game_plays = plays_by_game(week_df)
    
    # Pull games from dict
    games = week_game_plays.keys()
    
    # Create dataframe to hold results
    week_metrics = pd.DataFrame()
    
    # Loop through games and pull metrics
    for game in games:
 
        game_metrics = pd.DataFrame(game_analysis(pass_rushers_df, week_df, game, return_pursuit_angle = return_pursuit_angle))

        week_metrics = pd.concat([week_metrics, game_metrics])        

    return week_metrics 


def game_analysis(pass_rushers_df, week_df, game, return_pursuit_angle = True):
    '''
    Emits a dataframe with each pass rushers metrics for the game, along with the outcome from the scouting report.
    '''
    game_plays = plays_by_game(week_df)
    plays = game_plays[game]

    game_metrics = pd.DataFrame()

    for play in plays:

        # Added a try except since there are errors when the snap events are missing
        try:
            play_metrics = pd.DataFrame(play_analysis(pass_rushers_df, week_df, game, play, return_graph = False, return_pursuit_angle = return_pursuit_angle))

            game_metrics = pd.concat([game_metrics, play_metrics])
        
        except:
            print('Error loading (game, play):',game, play)

    return game_metrics


def play_analysis(pass_rushers_df, week_df, game, play, return_graph = True, return_pursuit_angle = True):
    '''
    Emits a dataframe with each pass rushers metric for the play, along with the outcome from the scouting report.
    '''
    play_pass_rush = get_play_pass_rushers(pass_rushers_df, game, play)

    # The data structure to hold the metric results
    play_metrics = pd.DataFrame()
    
    # The data structure to hold the graphing coordiantes
    graph_input = pd.DataFrame()
    
    # The data structure to hold the ids for the graph
    graph_ids = {}
    
    for pass_rusher in play_pass_rush.nflId:
        line_of_scrimmage, pass_rusher_analysis_frames, player_metrics = player_play_analysis(pass_rushers_df, week_df, pass_rusher, game, play, return_pursuit_angle = return_pursuit_angle)
        
        player_metrics = pd.DataFrame(player_metrics)
        
        play_metrics = pd.concat([play_metrics, player_metrics])
        
        graph_input = pd.concat([graph_input, pass_rusher_analysis_frames[['x','y']]])
        
        graph_ids[pass_rusher] = [pass_rusher_analysis_frames.x.iloc[0],pass_rusher_analysis_frames.y.iloc[0]]

    # Now get the ball data
    ball_graph_input = pass_rusher_analysis_frames[['ball_x','ball_y']]
    
    if return_graph == True:
        create_play_graph(line_of_scrimmage, graph_ids, graph_input, ball_graph_input)
    
    return play_metrics


def player_play_analysis(pass_rushers_df, week_df, pass_rusher, game, play, return_graph = False, return_pursuit_angle = True):
    '''
    Create a single player analysis
    '''
    # Isolate player
    play_pass_rushers = get_play_pass_rushers(pass_rushers_df, game, play)
    play_pass_rusher = play_pass_rushers[play_pass_rushers.nflId == pass_rusher]

    play_frames_df = get_play_frames(week_df, game, play)

    line_of_scrimmage, football_frames_df = play_fb_frames(play_frames_df)

    pass_rusher_analysis_frames = create_pass_rusher_analysis_frames(football_frames_df, play_frames_df, pass_rusher)

    pass_rusher_analysis_frames =  pass_rusher_game_play_metric_frames(pass_rusher_analysis_frames)

    metric = player_play_metric(pass_rusher_analysis_frames, return_pursuit_angle = return_pursuit_angle)

    if return_pursuit_angle == True:
            play_metric = metric[0]
            play_pursuit_angles = metric[1]

    else:
        play_metric = metric

    # Optional Graph
    if return_graph == True:
        create_single_player_graph(pass_rusher_analysis_frames, line_of_scrimmage)
    
    player_metrics = {'Player':play_pass_rusher.nflId,'Metrics':play_metric, 'Hit':play_pass_rusher.hit, 'Hurry':play_pass_rusher.hurry, 'Sack':play_pass_rusher.sack, 'Pressure':play_pass_rusher.pressure}

    if return_pursuit_angle == True:
        player_metrics['Pursuit Angle'] = play_pursuit_angles

    return line_of_scrimmage, pass_rusher_analysis_frames, player_metrics


# -----------------------------------------------------------------------------------------------------------------
# BUILD ANALYSIS DATAFRAMES
# -----------------------------------------------------------------------------------------------------------------

def get_play_pass_rushers(pass_rushers_df, game, play):
    '''
    Returns the the list of all pass rushers and their pressure stats for a given play.
    '''
    # Selects players who pass rushed on a given play
    play_pass_rushers_df = pass_rushers_df[pass_rushers_df.game == game][pass_rushers_df.play == play]

    return play_pass_rushers_df


def get_play_frames(week_df, game, play):
    '''
    Takes in the full weekly dataframe and returns selected play from a selected game in 0.1 second frames.
    '''
    # Extract frames for a given play
    play_frames_df = week_df[week_df.game == game][week_df.play == play]

    return play_frames_df


def play_fb_frames(play_frames_df):
    '''
    Isolates the football frames (football movement over the course of the play) for a given play.
    '''
    # nflId for the football is 0, creates a dataframe of the football over the play
    football_frames_df = play_frames_df[play_frames_df.nflId == 0].set_index('frame',drop = True)

    # Get the x value for the line of scrimmage (for use with play graphing)
    line_of_scrimmage = football_frames_df.x.iloc[0]

    # Remove all frames before snap and after event where ball is no longer being pass rushed
    snap_frame, end_frame = determine_pertinent_frames(football_frames_df)
    football_frames_df = football_frames_df[(football_frames_df.index >= snap_frame) & (football_frames_df.index <= end_frame)]

    # Clean the frames up
    football_frames_df = clean_fb_frames(football_frames_df)

    return line_of_scrimmage, football_frames_df


def create_pass_rusher_analysis_frames(football_frames_df, play_frames_df, pass_rusher):
    '''
    Merges football location data with player location on a 0.1s frame by frame basis data to compare the two to create the metrics
    '''
    # Create a dataframe which pairs the location and movement with the football and the pass risher, frame by frame (0.1 second intervals)
    pass_rusher_analysis_frames = football_frames_df.merge(play_frames_df[play_frames_df.nflId == pass_rusher].set_index('frame',drop = True),
                                        on = 'frame', how = 'left').reset_index(drop = True)

    return pass_rusher_analysis_frames


# -----------------------------------------------------------------------------------------------------------------
# DATAFRAME BUILDING SUPPORT FUNCTIONS
# -----------------------------------------------------------------------------------------------------------------

def determine_pertinent_frames(football_frames):
    '''
    This function determines the indices of the relevant frames for pass rush analysis.
    Returns the starting (snap) index and the ending index for the analysis.  
    '''
    # Initiate a trigger which tells the function to start checking the events after ball snap for the end event (when qb passes, is sacked, etc.)
    trigger = 0
    
    for i, event in enumerate(football_frames.event):
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
                return snap_index, i + 1
        else:
            continue

def clean_fb_frames(football_frames):
    '''
    Cleans up the football tracking df
    '''
    football_frames = football_frames.drop(columns = ['game',
                                        'play',
                                        'nflId',
                                        'o',
                                        'dir',
                                        'event']).rename(columns = {'x':'ball_x',
                                                                    'y':'ball_y',
                                                                    's':'ball_s',
                                                                    'a':'ball_a',
                                                                    'dis':'ball_dis'})
    
    return football_frames


# -----------------------------------------------------------------------------------------------------------------
#  METRICS CREATION FUNCTION
# -----------------------------------------------------------------------------------------------------------------

def player_play_metric(pass_rusher_analysis_frames, return_pursuit_angle = False):
    '''
    Emits the player's metric for the play.  Also emits the pursuit angle, which itself *might* be indicative of something...
    '''
    if return_pursuit_angle == True:
        return pass_rusher_analysis_frames.metric.mean(), pass_rusher_analysis_frames.pursuit_factor.mean()
    
    else:
        return round(pass_rusher_analysis_frames.metric.mean(),3)


def pass_rusher_game_play_metric_frames(pass_rusher_analysis_frames):
    '''
    Creates the metric for a given pass rusher for a given play in a given game.  Returns the metric and dataframe if desired.
    '''

    # The next four lines use metric support functions below
    pass_rusher_analysis_frames = get_distances(pass_rusher_analysis_frames)

    pass_rusher_analysis_frames = add_prev_coord(pass_rusher_analysis_frames)

    pass_rusher_analysis_frames = find_pursuit_angle(pass_rusher_analysis_frames)

    pass_rusher_analysis_frames = find_escape_angle(pass_rusher_analysis_frames)

    # Clean things up by removing first and last frame
    pass_rusher_analysis_frames = pass_rusher_analysis_frames[1:-1]

    pass_rusher_analysis_frames = true_pursuit(pass_rusher_analysis_frames)

    # Drop unecessary columns
    pass_rusher_analysis_frames = pass_rusher_analysis_frames.drop(columns = ['ball_s','ball_a','game','play','s','a','o','dir','event'])

    # ! *********The following is the current metric and can be replaced**********
    pass_rusher_analysis_frames = metric_calculation(pass_rusher_analysis_frames)
    # ! **************************************************************************

    return pass_rusher_analysis_frames


def metric_calculation(pass_rusher_analysis_frames):
    pass_rusher_analysis_frames['metric'] = (pass_rusher_analysis_frames.prev_distance + pass_rusher_analysis_frames.true_pursuit)/(pass_rusher_analysis_frames.ball_player_distance)

    return pass_rusher_analysis_frames


# -----------------------------------------------------------------------------------------------------------------
#  METRICS SUPPORT FUNCTIONS
# -----------------------------------------------------------------------------------------------------------------

def get_distances(pass_rusher_analysis_frames):
    '''
    Gets current frame distance as well as adds previous frame distance to dataframe
    '''
    # Simply use pythagorean formula to calculate distance at time of frame based on x and y coordinates
    pass_rusher_analysis_frames['ball_player_distance'] = ((pass_rusher_analysis_frames.x - pass_rusher_analysis_frames.ball_x)**2 +
                                (pass_rusher_analysis_frames.y - pass_rusher_analysis_frames.ball_y)**2)**.5

    # Since the previous locations are required for the calculation of the metric later on, create a column with previous distance
    pass_rusher_analysis_frames['prev_distance'] = pass_rusher_analysis_frames.ball_player_distance.shift(1)

    return pass_rusher_analysis_frames


def add_prev_coord(pass_rusher_analysis_frames):
    '''
    Adds previous coordinates of ball and pass rusher to dataframe
    '''  
    # Shifting the x and y allows to compare past coordinate location to present, giving us a movement vector
    pass_rusher_analysis_frames['shift_x'] = pass_rusher_analysis_frames.x.shift(1)
    pass_rusher_analysis_frames['shift_y'] = pass_rusher_analysis_frames.y.shift(1)

    # Shifting the x and y for the ball as well to get its movement vector
    pass_rusher_analysis_frames['shift_ball_x'] = pass_rusher_analysis_frames.ball_x.shift(1)
    pass_rusher_analysis_frames['shift_ball_y'] = pass_rusher_analysis_frames.ball_y.shift(1)

    return pass_rusher_analysis_frames


def find_pursuit_angle(pass_rusher_analysis_frames):
    '''
    Calcualtes the angle between the player and the ball after they both move, and the direction the player actually 
    moved in that interval.  A negative value means the player is moving in a direction away from the ball; a zero (0) 
    value means they are moving perpendicular to it; for positive values, the closer to one the more direct the movement -
    a value of one means they move directly towards the ball in that interval.
    '''
    # Need to initialize with a value to account for the shift, this frame will later be dropped
    pursuit_angle_factor = [[0,0]]

    for i in pass_rusher_analysis_frames.index[1:]:
        pursuit_angle_factor.append(
            1 - spatial.distance.cosine(
                [round(pass_rusher_analysis_frames.x.iloc[i]-pass_rusher_analysis_frames.shift_x.iloc[i],3),
                round(pass_rusher_analysis_frames.y.iloc[i]-pass_rusher_analysis_frames.shift_y.iloc[i],3)],
                [round(pass_rusher_analysis_frames.ball_x.iloc[i]-pass_rusher_analysis_frames.shift_x.iloc[i],3),
                round(pass_rusher_analysis_frames.ball_y.iloc[i]-pass_rusher_analysis_frames.shift_y.iloc[i],3)]))

    # Add this list as a dataframe column
    pass_rusher_analysis_frames['pursuit_factor'] = pursuit_angle_factor

    return pass_rusher_analysis_frames


def find_escape_angle(pass_rusher_analysis_frames):
    '''
    Similar to the pursuit angle, calculates how the ball is moving in relation to the pass rusher
    '''
    # Need to initialize with a value to account for the shift, this frame will later be dropped
    escape_angle_factor = [[0,0]]

    for i in pass_rusher_analysis_frames.index[1:]:
        escape_angle_factor.append(
            1 - spatial.distance.cosine(
                [round(pass_rusher_analysis_frames.ball_x.iloc[i]-pass_rusher_analysis_frames.shift_ball_x.iloc[i],3),
                round(pass_rusher_analysis_frames.ball_y.iloc[i]-pass_rusher_analysis_frames.shift_ball_y.iloc[i],3)],
                [round(pass_rusher_analysis_frames.x.iloc[i]-pass_rusher_analysis_frames.shift_ball_x.iloc[i],3),
                round(pass_rusher_analysis_frames.y.iloc[i]-pass_rusher_analysis_frames.shift_ball_y.iloc[i],3)]))

    # Add this list as a dataframe column
    pass_rusher_analysis_frames['escape_factor'] = escape_angle_factor

    return pass_rusher_analysis_frames


def true_pursuit(pass_rusher_analysis_frames):
    '''
    This function combines the effects of 'pursuit' and 'escape' to see how the rusher is pursuing the ball, independent
    of the quarterback's movement.
    '''
    pass_rusher_analysis_frames['pursuit'] = pass_rusher_analysis_frames.dis * pass_rusher_analysis_frames.pursuit_factor
    pass_rusher_analysis_frames['escape'] = pass_rusher_analysis_frames.ball_dis * -pass_rusher_analysis_frames.escape_factor

    pass_rusher_analysis_frames['true_pursuit'] = pass_rusher_analysis_frames.pursuit + pass_rusher_analysis_frames.escape

    return pass_rusher_analysis_frames

# -----------------------------------------------------------------------------------------------------------------
#  GRAPHING FUNCTIONS
# -----------------------------------------------------------------------------------------------------------------

def create_play_graph(line_of_scrimmage, graph_ids, graph_input, ball_graph_input):
    '''
    Creates a graph of all pass rushers on a given play
    '''
     # Import graphics libraries
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Create figure
    plt.figure(figsize = [9,9])
    plt.title('---- Pass Rusher (red) vs. Ball (green) by Frame ----\nFrames go from Light (snap) to Dark (end of play)', fontsize = 16)
    plt.xlabel('Absolute Yardline')
    plt.ylabel('')
    
    # Create and graph the datapoints
    sns.scatterplot(x = ball_graph_input.ball_x, y = ball_graph_input.ball_y, hue = ball_graph_input.index, palette = 'Greens', legend = False)
    sns.scatterplot(x = graph_input.x, y = graph_input.y, hue = graph_input.index, palette = 'Reds', legend = False)
   
    # Create the frame labels for the players
    for mark in zip(graph_input.index,graph_input.x,graph_input.y):
        plt.annotate(mark[0],
                    (mark[1], mark[2]),
                    textcoords = 'offset points',
                    xytext = (4,4),
                    ha = 'center',
                    fontsize = 8)

    # Create the frame labels for the football
    for mark in zip(ball_graph_input.index,ball_graph_input.ball_x,ball_graph_input.ball_y):
        plt.annotate(mark[0],
                    (mark[1], mark[2]),
                    textcoords = 'offset points',
                    xytext = (4,4),
                    ha = 'center',
                    fontsize = 8)
        
    # Create the labels to identify which rusher is whom
    for identity in graph_ids:
        plt.annotate(identity,
                    (graph_ids[identity][0],graph_ids[identity][1]),
                    textcoords = 'offset points',
                    xytext = (-8,-4),
                    ha = 'right',
                    fontsize = 12)
 
    # At figsize = [12,12], the line width of the line of scrimmage = 19.2; change in proportion to this
    plt.axvline(x = line_of_scrimmage,
                lw = 14.4,
                alpha = 0.2)
    plt.axvline(x = line_of_scrimmage)
    plt.text(line_of_scrimmage, (ball_graph_input.ball_y.mean() + graph_input.y.mean())/2, 
            f'Line of Scrimmage\n~{int(line_of_scrimmage)} yard line',
            fontsize = 15, rotation = 90,
            ha = 'center',
            va = 'top')

    plt.show()


def create_single_player_graph(pass_rusher_analysis_frames, line_of_scrimmage):
    '''
    Creates a graph of the ball and a single pass rusher
    '''
    # Import graphics libraries
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Create figure
    plt.figure(figsize = [9,9])
    plt.title('---- Pass Rusher (red) vs. Ball (green) by Frame ----\nFrames go from Light (snap) to Dark (end of play)', fontsize = 16)
    plt.xlabel('Absolute Yardline')
    plt.ylabel('')
    
    # Create and graph the datapoints
    sns.scatterplot(x = pass_rusher_analysis_frames.ball_x, y = pass_rusher_analysis_frames.ball_y, hue = pass_rusher_analysis_frames.index, palette = 'Greens', legend = False)
    sns.scatterplot(x = pass_rusher_analysis_frames.x, y = pass_rusher_analysis_frames.y, hue = pass_rusher_analysis_frames.index, palette = 'Reds', legend = False)
   
    # Create the frame labels for the player
    for mark in zip(pass_rusher_analysis_frames.index,pass_rusher_analysis_frames.x,pass_rusher_analysis_frames.y):
        plt.annotate(mark[0],
                    (mark[1], mark[2]),
                    textcoords = 'offset points',
                    xytext = (4,4),
                    ha = 'center',
                    fontsize = 8)

    # Create the frame labels for the football
    for mark in zip(pass_rusher_analysis_frames.index,pass_rusher_analysis_frames.ball_x,pass_rusher_analysis_frames.ball_y):
        plt.annotate(mark[0],
                    (mark[1], mark[2]),
                    textcoords = 'offset points',
                    xytext = (4,4),
                    ha = 'center',
                    fontsize = 8)
 
    # At figsize = [12,12], the line width of the line of scrimmage = 19.2; change in proportion to this
    plt.axvline(x = line_of_scrimmage,
                lw = 14.4,
                alpha = 0.2)
    plt.axvline(x = line_of_scrimmage)
    plt.text(line_of_scrimmage, (pass_rusher_analysis_frames.ball_y.mean() + pass_rusher_analysis_frames.y.mean())/2, 
            f'Line of Scrimmage\n~{int(line_of_scrimmage)} yard line',
            fontsize = 15, rotation = 90,
            ha = 'center',
            va = 'top')

    plt.show()


# -----------------------------------------------------------------------------------------------------------------
#  TESTING FUNCTIONS
# -----------------------------------------------------------------------------------------------------------------

def random_play(week1_df):
    '''
    Creates a random play to analyze using week 1 plays
    '''
    week_game_plays = []
    for game in week1_df.game.unique():
        for play in week1_df[week1_df.game == game].play.unique():
            week_game_plays.append((game,play))
    
    game_index = np.random.randint(0,1175)

    game = week_game_plays[game_index][0]
    play = week_game_plays[game_index][1]

    return game, play


# -----------------------------------------------------------------------------------------------------------------
#  SUPPORT FUNCTIONS
# -----------------------------------------------------------------------------------------------------------------
def plays_by_game(week_df):
    '''
    Returns a dictionary with the keys being the games of the week and the values being a list of plays
    '''
    game_plays = {}
    for game in week_df.game.unique():
        plays = []
        for play in week_df[week_df.game == game].play.unique():
            plays.append(play)
        game_plays[game] = plays

    return game_plays

def group_results(metrics_df):
    '''
    
    '''
    play_count = list(metrics_df.groupby(by = 'Player').Player.count())

    metrics_df = metrics_df.groupby(by = 'Player').mean()

    metrics_df['play_count'] = play_count

    return metrics_df
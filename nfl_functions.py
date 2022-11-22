import numpy as np
import pandas as pd

from scipy import spatial

'''
The following functions are built to create the Chase Metric for various levels of analysis (play, game, season)
'''

# -----------------------------------------------------------------------------------------------------------------
# MASTER FUNCTIONS
# -----------------------------------------------------------------------------------------------------------------

def analysis_dataframe():
    '''
    This would be a dataframe where each column was a week, and each row a player.  Would then have a total.
    '''


# -----------------------------------------------------------------------------------------------------------------
# ACQUIRE AND PREPARE DATA
# -----------------------------------------------------------------------------------------------------------------

def prep_scout_data(scout_csv):
    '''
    Takes in NFL PFF scouting data (in csv form) and distills it into dataframe of pass rushers and their outcomes.
    '''
    # Pull in data
    scout = pd.read_csv(scout_csv)

    # Isolate pass rushers
    pass_rushers = scout[scout.pff_role == 'Pass Rush']

    # Strip unnecessary columns, rename and retype
    pass_rushers = pass_rushers[['gameId',
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

    # Create a feature indicating if there was a hit, hurry or sack
    # !-In future, look into weighing these
    pass_rushers['pressure'] = pass_rushers.hit + pass_rushers.hurry + pass_rushers.sack

    return pass_rushers


def prep_weekly_data(week_csv):
    '''
    Takes in weekly player tracking data (in csv form) and distills it into dataframe for further analysis.
    '''
    # Pull in data
    week = pd.read_csv(week_csv)

    # Strip unecessary columns, rename fill NaNs (all for the 'football' and retype)
    week = week.drop(columns = ['time','playDirection','team','jerseyNumber']).rename(columns = {'gameId':'game',
                                                                        'playId':'play',
                                                                        'frameId':'frame',}).fillna(0).astype({'nflId':int})
    
    return week


# -----------------------------------------------------------------------------------------------------------------
# BUILD ANALYSIS DATAFRAMES
# -----------------------------------------------------------------------------------------------------------------

def get_play_pass_rushers(pass_rushers_df, game, play):
    '''
    Returns the the list of all pass rushers for a given play.
    '''
    # Selects players who pass rushed on a given play
    play_pass_rush = pass_rushers_df[pass_rushers_df.game == game][pass_rushers_df.play == play]

    # Makes a list of pass rushers by NFL ID
    pass_rush_list = list(play_pass_rush.nflId)

    return pass_rush_list


def get_play_frames(week_df, game, play):
    '''
    Takes in the full weekly dataframe and returns selected play from a selected game in 0.1 second frames.
    '''
    # Extract frames for a given play
    play_frames = week_df[week_df.game == game][week_df.play == play]

    return play_frames


def play_fb_frames(play_frames_df):
    '''
    Isolates the football frames (football movement over the course of the play) for a given play.
    '''
    # nflId for the football is 0, creates a dataframe of the football over the play
    football_frames = play_frames_df[play_frames_df.nflId == 0].set_index('frame',drop = True)

    # Get the x value for the line of scrimmage (for use with play graphing)
    line_of_scrimmage = football_frames.x.iloc[0]

    # Remove all frames before snap and after event where ball is no longer being pass rushed
    snap_frame, end_frame = determine_pertinent_frames(football_frames)
    football_frames = football_frames[(football_frames.index >= snap_frame) & (football_frames.index <= end_frame)]

    # Clean the frames up
    football_frames = clean_fb_frames(football_frames)

    return line_of_scrimmage, football_frames


def create_pass_rusher_analysis_frames(football_frames, play_frames, pass_rusher):
    '''
    Merges football location data with player location on a 0.1s frame by frame basis data to compare the two to create the metrics
    '''
    # Create a dataframe which pairs the location and movement with the football and the pass risher, frame by frame (0.1 second intervals)
    pass_rusher_analysis_frames = football_frames.merge(play_frames[play_frames.nflId == pass_rusher].set_index('frame',drop = True),
                                        on = 'frame', how = 'left').reset_index(drop = True)

    return pass_rusher_analysis_frames

# -----------------------------------------------------------------------------------------------------------------
# DATAFRAME BUILD SUPPORT FUNCTIONS
# -----------------------------------------------------------------------------------------------------------------

def determine_pertinent_frames(football_frames):
    '''
    This function determines the indices of the relevant frames for pass rush analysis.
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
#  METRICS FUNCTIONS
# -----------------------------------------------------------------------------------------------------------------

def metrics(pass_rusher_analysis_frames, return_graph = False, line_of_scrimmage = 0):
    '''
    Creates the metric features and intermediate columns to get to them.  Returns the metrics.
    '''
    # Determine distance between the football and the pass rusher for each frame using the pythagorean formula.
    pass_rusher_analysis_frames['distance'] = ((pass_rusher_analysis_frames.x - pass_rusher_analysis_frames.ball_x)**2 +
                                (pass_rusher_analysis_frames.y - pass_rusher_analysis_frames.ball_y)**2)**.5

    # Determine the change of distance between football player and ball from frame to frame.
    # Note: negative number means player and ball are closer together at frame.
    pass_rusher_analysis_frames['change_distance'] = pass_rusher_analysis_frames.distance.diff()  

    # Shifting the x and y for the player allows to compare past coordiante location to present, giving us a movement vector
    pass_rusher_analysis_frames['shift_x'] = pass_rusher_analysis_frames.x.shift(1)
    pass_rusher_analysis_frames['shift_y'] = pass_rusher_analysis_frames.y.shift(1)

    # Call pursuit angle function
    pass_rusher_analysis_frames = find_pursuit_angle(pass_rusher_analysis_frames)

    # Clean things up by removing first and last frame and dropping unecessary columns; lost data relatively unimpactful
    pass_rusher_analysis_frames = pass_rusher_analysis_frames[1:-1]

    # Optional Graph
    if return_graph == True:
        create_graph(pass_rusher_analysis_frames, line_of_scrimmage)

    # ***This is probably not necessary for production function***
    pass_rusher_analysis_frames = pass_rusher_analysis_frames.drop(columns = ['ball_s','ball_a','game','play','s','a','o','dir','event'])

    # !-The following is the current metric and can be replaced
    pass_rusher_analysis_frames['test_factor_a'] = ((pass_rusher_analysis_frames.pursuit_angle * pass_rusher_analysis_frames.dis) - 
                                                    pass_rusher_analysis_frames.change_distance)/pass_rusher_analysis_frames.distance

    return pass_rusher_analysis_frames.pursuit_angle.mean(), pass_rusher_analysis_frames.test_factor_a.mean()

# -----------------------------------------------------------------------------------------------------------------
#  METRICS SUPPORT FUNCTIONS
# -----------------------------------------------------------------------------------------------------------------

def find_pursuit_angle(pass_rusher_analysis_frames):
    '''
    I am sure there is a way to calcualte directly in df using assign, but this calcualtes the angle between the player
    and the ball after they both move, and the direction the player actually moved in that interval.  A negative value
    means the player is moving in a direction away from the ball; a zero (0) value means they are moving perpendicular
    to it; for positive values, the closer to one the more direct the movement - a value of one means they move directly
    towards the ball in that interval.
    '''
    # Need to initialize with a value to account for the shift
    cosine_factor = [[0,0]]

    # Create vectors with the tails at a player's location in frame 'i', with the heads at the player location and ball 
    # location in frame 'i+1' 
    for i in pass_rusher_analysis_frames.index[1:]:
        cosine_factor.append(
            1 - spatial.distance.cosine(
                [round(pass_rusher_analysis_frames.x.iloc[i]-pass_rusher_analysis_frames.shift_x.iloc[i],3),
                round(pass_rusher_analysis_frames.y.iloc[i]-pass_rusher_analysis_frames.shift_y.iloc[i],3)],
                [round(pass_rusher_analysis_frames.ball_x.iloc[i]-pass_rusher_analysis_frames.shift_x.iloc[i],3),
                round(pass_rusher_analysis_frames.ball_y.iloc[i]-pass_rusher_analysis_frames.shift_y.iloc[i],3)]))

    pass_rusher_analysis_frames['pursuit_angle'] = cosine_factor

    return pass_rusher_analysis_frames


def create_graph(pass_rusher_analysis_frames, line_of_scrimmage):
    '''
    Creates a simple graph of the ball and the ball
    '''
    # Import graphics libraries
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Create figure
    plt.figure(figsize = [9,9])
    plt.title('---- Pass Rusher (red) vs. Ball (green) by Frame ----\nFrames go from Light (snap) to Dark (end of play)', fontsize = 16)
    plt.xlabel('Absolute Yardline')
    plt.ylabel('')
    
    # Create and graphe the datapoints
    sns.scatterplot(x = pass_rusher_analysis_frames.ball_x, y = pass_rusher_analysis_frames.ball_y, hue = pass_rusher_analysis_frames.index, palette = 'Greens', legend = False)
    sns.scatterplot(x = pass_rusher_analysis_frames.x, y = pass_rusher_analysis_frames.y, hue = pass_rusher_analysis_frames.index, palette = 'Reds', legend = False)
   
    # Create the frame lables for the player
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

def random_play(week):
    '''
    Creates a random play to analyze
    '''
    week_game_plays = []
    for game in week.game.unique():
        for play in week[week.game == game].play.unique():
            week_game_plays.append((game,play))
    
    game_index = np.random.randint(0,1175)

    game = week_game_plays[game_index][0]
    play = week_game_plays[game_index][1]

    return game, play
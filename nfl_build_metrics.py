import nfl_frame_builder as nfl
import nfl_acquire_and_prep as acquire

import pandas as pd
import numpy as np
import itertools
from scipy import spatial
import re

import warnings
warnings.filterwarnings('ignore')

# ====================================================================================================
# ENHANCE FRAME AND ADD METRICS
# ====================================================================================================

def build_metrics(analysis_frames, point_of_scrimmage, players_df):
    '''
    Combines all functions to take a given playframe and return frame by frame metrics 
    
    Parameters:
        'analysis_frames' - Dataframe - Complete frames of the play
        'point_of_scrimmage' - Tuple of Floats - x and y coordinates of snap
        'players_df' - Dataframe - Contains player information, including weight
    Returns:
        'analysis_frames' - Dataframe - Complete frames of the play with all metric stuff added
    '''
    analysis_frames = recenter_on_snap(point_of_scrimmage, analysis_frames)
    analysis_frames = create_movement_vectors(analysis_frames)
    analysis_frames = create_rusher_to_ball_vector(analysis_frames)
    analysis_frames = create_ball_to_rusher_vector(analysis_frames)
    analysis_frames = create_change_in_distance_measurement(analysis_frames)
    analysis_frames = create_change_in_distance_ratio(analysis_frames)
    analysis_frames = player_opponent_distance(analysis_frames)
    analysis_frames = add_pass_rusher_to_ball_colinearity(analysis_frames)
    analysis_frames = create_pursuit_factor(analysis_frames)
    analysis_frames = pass_rusher_to_ball_vectors(analysis_frames)
    analysis_frames = create_escape_factor(analysis_frames)
    analysis_frames = ball_to_pass_rusher_vectors(analysis_frames)
    analysis_frames = pass_rusher_force(analysis_frames, players_df)
    analysis_frames = create_true_pursuit(analysis_frames)
    analysis_frames = create_pursuit1(analysis_frames)
    analysis_frames = create_pursuit2(analysis_frames)
    analysis_frames = create_pursuit3(analysis_frames)
    analysis_frames = create_pursuit4(analysis_frames)

    return analysis_frames


# ----- Sub Functions -----------------------------------------------------------------------------

def recenter_on_snap(point_of_scrimmage, analysis_frames):
    '''
    Applies the reorientate_coord function to all x and y coordinates in dataframe.
    
    Parameters:
        'point_of_scrimmage' - Tuple of Floats - x and y coordinates of snap
        'analysis_frames' - Dataframe - Complete focused frames of the play ready for analysis
    Returns:
        'analysis_frames' - Dataframe - Complete focused frames of the play ready for analysis - now reoriented
    '''
    # Create the x and y regex patters to match
    pattern_x = r'.*[x]$'
    pattern_y = r'.*[y]$'

    # Seperate the point of scrimmage/origin into x and y coordinates
    origin_x = point_of_scrimmage[0]
    origin_y = point_of_scrimmage[1]

    # Look through analysis frames column names to identify whcih ones need to be reoriented to new x or y
    for col in analysis_frames.columns:
        if re.match(pattern_x, col):
            analysis_frames[col] = reorientate_coord(origin_x, analysis_frames[col])
        if re.match(pattern_y, col):
            analysis_frames[col] = reorientate_coord(origin_y, analysis_frames[col])

    return analysis_frames


def create_movement_vectors(analysis_frames):
    '''
    Finds the x- and y-components of the movement vector for all players and ball, that is the vector between the
    current frame's location and the next frame's location
    
    Parameters:
        'analysis_frames' - Dataframe - Complete frames of the play ready
    Returns:
        'analysis_frames' - Dataframe - Complete frames of the play, now with movement vectors added
    '''
    # Look through the columns and find those which match pass rusher or blockers
    for column in analysis_frames.columns:
        match = re.match(r'(ball|pass_rusher|pass_blocker)(_?\d?_)(x|y)', column)
        if match:
            # Use the groups in the matched regex expression to create new column
            mvmt_type = match.group(1)
            connector = match.group(2)
            coord_type = match.group(3)
            original_coord = f"{mvmt_type}{connector}{coord_type}"
            next_coord = f"{mvmt_type}{connector}next_{coord_type}"
            # Subtract the original coordinate from the next coordinate
            analysis_frames[f'{mvmt_type}{connector}distance_moved_{coord_type}'] = analysis_frames[next_coord] - analysis_frames[original_coord]
            
    return analysis_frames


def create_rusher_to_ball_vector(analysis_frames):
    '''
    Create a vector that represents the distance from the player's CURRENT position to the ball's NEXT position.
    
    Parameters:
        'analysis_frames' - Dataframe - Complete frames of the play
    Returns:
        'analysis_frames' - Dataframe - Complete frames of the play ready for analysis with added vector components
    '''
    # Use vector component subtraction 
    analysis_frames['pass_rusher_to_ball_vector_x'] = analysis_frames.ball_next_x - analysis_frames.pass_rusher_x
    analysis_frames['pass_rusher_to_ball_vector_y'] = analysis_frames.ball_next_y - analysis_frames.pass_rusher_y
    
    return analysis_frames


def create_ball_to_rusher_vector(analysis_frames):
    '''
    Create a vector that represents the distance between the ball's CURRENT position and the player's NEXT position.
    
    Parameters:
        'analysis_frames' - Dataframe - Complete frames of the play ready for analysis
    Returns:
        'analysis_frames' - Dataframe - Complete frames of the play ready for analysis with added vector components
    '''
    # Use vector component subtraction 
    analysis_frames['ball_to_pass_rusher_vector_x'] = analysis_frames.ball_next_x - analysis_frames.pass_rusher_next_x
    analysis_frames['ball_to_pass_rusher_vector_y'] = analysis_frames.ball_next_y - analysis_frames.pass_rusher_next_y
    
    return analysis_frames


def create_change_in_distance_measurement(analysis_frames):
    '''
    Find the change in distance between the player and the ball (next_xy and xy)
    
    Parameters:
        'analysis_frames' - Dataframe - Complete frames of the play ready for analysis
    Returns:
        'analysis_frames' - Dataframe - Complete frames of the play ready for analysis with analysis with change of distances
    '''   
    change_in_pass_rusher_to_ball_dist_list = []
    
    for i in analysis_frames.index:
        n = round(((analysis_frames.ball_next_x.iloc[i] - analysis_frames.pass_rusher_next_x.iloc[i])**2 +
                                                        (analysis_frames.ball_next_y.iloc[i] - analysis_frames.pass_rusher_next_y.iloc[i])**2)**.5 - 
                                                       ((analysis_frames.ball_x.iloc[i] - analysis_frames.pass_rusher_x.iloc[i])**2 +
                                                        (analysis_frames.ball_y.iloc[i] - analysis_frames.pass_rusher_y.iloc[i])**2)**.5, 4)
        if n > .0001:
            change_in_pass_rusher_to_ball_dist_list.append(n)
        else:
            change_in_pass_rusher_to_ball_dist_list.append(.0001)
    
    analysis_frames['change_in_pass_rusher_to_ball_dist'] = change_in_pass_rusher_to_ball_dist_list
    
    return analysis_frames


def create_change_in_distance_ratio(analysis_frames):
    '''
    Find the change in distance between the player and the ball (next_xy and xy)
    
    Parameters:
        'analysis_frames' - Dataframe - Complete frames of the play ready for analysis
    Returns:
        'analysis_frames' - Dataframe - Complete frames of the play ready for analysis with analysis with change of distances
    '''   
    change_in_pass_rusher_to_ball_dist_ratio_list = []
    
    for i in analysis_frames.index:
        n = round(((analysis_frames.ball_x.iloc[i] - analysis_frames.pass_rusher_x.iloc[i])**2 +
                   (analysis_frames.ball_y.iloc[i] - analysis_frames.pass_rusher_y.iloc[i])**2)**.5 /
                  ((analysis_frames.ball_next_x.iloc[i] - analysis_frames.pass_rusher_next_x.iloc[i])**2 +
                   (analysis_frames.ball_next_y.iloc[i] - analysis_frames.pass_rusher_next_y.iloc[i])**2)**.5, 4)
        
        change_in_pass_rusher_to_ball_dist_ratio_list.append(n)
    
    analysis_frames['pass_rusher_to_ball_dist_ratio'] = change_in_pass_rusher_to_ball_dist_ratio_list
    
    return analysis_frames


def player_opponent_distance(analysis_frames):
    '''
    For each opponent (blocker or rusher) adds the distance between them and the analyzed player to the analysis frame.
    Note: Only for v_type = PvP with opponents
    
    Parameters:
        'analysis_frames' - Dataframe - Complete frames of the play ready for analysis
    Returns:
        'analysis_frames' - Dataframe - Complete frames of the play ready for analysis with added distances
    '''
    # Use internal variables to make this visually easier to follow
    prx1 = analysis_frames.pass_rusher_x
    pry1 = analysis_frames.pass_rusher_y
    
    # Get the pass blocker column names
    pass_blocker_list = get_pass_blockers(analysis_frames)
    
    for pass_blocker in pass_blocker_list:
        
        pbx1 = analysis_frames[f'{pass_blocker}_x']
        pby1 = analysis_frames[f'{pass_blocker}_y']
        
        # Get distance from player to opponent
        analysis_frames['temp'] = round(euclidean_distance(prx1, pry1, pbx1, pby1),4)
        
        # Rename the columns
        analysis_frames = analysis_frames.rename(columns = {'temp':f'pass_rusher_{pass_blocker}_dist'})
        
    return analysis_frames


def add_pass_rusher_to_ball_colinearity(analysis_frames):
    '''
    Computes how similar (colinear) the movemens are between the player and the ball.
    
    Parameters:
        'analysis_frames' - Dataframe - Complete frames of the play
    Returns:
        'analysis_frames' - Dataframe - Complete frames of the play with colinearity added

    '''
    # initialize the list holding the frame by frame pursuit angle calculation
    colinearity_list = []

    for i in analysis_frames.index:
        # Calculate pursuit angle
        colinearity_list.append(round(pass_rusher_to_ball_colinearity_calc(analysis_frames.pass_rusher_distance_moved_x.iloc[i],
                                                                     analysis_frames.pass_rusher_distance_moved_y.iloc[i],
                                                                     analysis_frames.ball_distance_moved_x.iloc[i],
                                                                     analysis_frames.ball_distance_moved_y.iloc[i]),4))

    analysis_frames['colinearity'] = colinearity_list
    
    return analysis_frames


def create_pursuit_factor(analysis_frames):
    '''
    Calculates the 'pursuit factor', which is a ratio (from -1 to 1) based on the angle between the pass rusher's
    movement vector, and the vector from the pass rusher to the ball after it moves.
    A negative value means the player is moving in a direction away from the ball; a zero (0) value means they are 
    moving perpendicular to it; for positive values, the closer to one the more direct the movement -
    a value of one means they move directly towards the ball in that interval.
    
    Parameters:
        'analysis_frames' - Dataframe - Complete frames of the play
    Returns:
        'analysis_frames' - Dataframe - Complete frames of the play with pursuit factor added
    '''
    # initialize the list holding the frame by frame pursuit angle calculation
    pursuit_factor_list = []
    
    # This always compares pass rusher to ball
    for i in analysis_frames.index:
        # Calculate pursuit angle
        pursuit_factor_list.append(round(pursuit_factor_calc(analysis_frames.pass_rusher_to_ball_vector_x.iloc[i],
                                                       analysis_frames.pass_rusher_to_ball_vector_y.iloc[i],
                                                       analysis_frames.pass_rusher_distance_moved_x.iloc[i],
                                                       analysis_frames.pass_rusher_distance_moved_y.iloc[i]),4))

    analysis_frames['pursuit_factor'] = pursuit_factor_list

    return analysis_frames


def pass_rusher_to_ball_vectors(analysis_frames):
    '''
    Finds the component of the player's movement vector that is going towards the ball.
    
    Parameters:
        'analysis_frames' - Dataframe - Complete frames of the play
    Returns:
        'analysis_frames' - Dataframe - Complete frames of the play with vector components towards ball added
    '''
    # Create lists to hold series to be added to dataframe
    vec_x = []
    vec_y = []
    
    for i in analysis_frames.index:
        p_p_x = analysis_frames.pass_rusher_distance_moved_x[i]
        p_p_y = analysis_frames.pass_rusher_distance_moved_y[i]

        p_b_x = analysis_frames.pass_rusher_to_ball_vector_x[i]
        p_b_y = analysis_frames.pass_rusher_to_ball_vector_y[i]
    
        # Find the x and y components of the player movement relative to the ball
        components = component(p_p_x, p_p_y, p_b_x, p_b_y)

        vec_x.append(round(components[0],4))
        vec_y.append(round(components[1],4))

    # This rewrites the analysis_frames over
    analysis_frames['pass_rusher_to_ball_vector_x'] = vec_x
    
    analysis_frames['pass_rusher_to_ball_vector_y'] = vec_y

    return analysis_frames


def create_escape_factor(analysis_frames):
    '''
    Similar to the pursuit angle, calculates how the ball is moving in relation to the pass rusher.
    
    Parameters:
        'analysis_frames' - Dataframe - Complete frames of the play
    Returns:
        'analysis_frames' - Dataframe - Complete frames of the play with escape factor added
    '''
    # Need to initialize with a value to account for the shift, this frame will later be dropped
    escape_factor_list = []

    for i in analysis_frames.index:
        # Calculate pursuit angle
        escape_factor_list.append(round(escape_factor_calc(analysis_frames.ball_distance_moved_x.iloc[i],
                                                     analysis_frames.ball_distance_moved_y.iloc[i],
                                                     analysis_frames.ball_to_pass_rusher_vector_x.iloc[i],
                                                     analysis_frames.ball_to_pass_rusher_vector_y.iloc[i]),4))

    analysis_frames['escape_factor'] = escape_factor_list
    
    return analysis_frames


def ball_to_pass_rusher_vectors(analysis_frames):
    '''
    Finds the component of the ball's movement vector that is going away from the player.
    
    Parameters:
        'analysis_frames' - Dataframe - Complete frames of the play
    Returns:
        'analysis_frames' - Dataframe - Complete frames of the play with vector components away from player added
    '''
    vec_x = []
    vec_y = []
    
    for i in analysis_frames.index:
        b_b_x = analysis_frames.ball_distance_moved_x[i]
        b_b_y = analysis_frames.ball_distance_moved_y[i]

        b_p_x = analysis_frames.ball_to_pass_rusher_vector_x[i]
        b_p_y = analysis_frames.ball_to_pass_rusher_vector_y[i]

        components = component(b_b_x, b_b_y, b_p_x, b_p_y)

        vec_x.append(round(components[0],4))
        vec_y.append(round(components[1],4))

    # This conveniently rewrites the analysis_frames over
    analysis_frames['ball_to_pass_rusher_vector_x'] = vec_x
    
    analysis_frames['ball_to_pass_rusher_vector_y'] = vec_y

    return analysis_frames


def pass_rusher_force(analysis_frames, players_df):
    '''
    Integrates the weight of the pass rusher to find the force they are applying in the direction of the ball 
    during pursuit.
    F = mass * acceleration * pursuit factor
    
    Parameters:
        'analysis_frames' - Dataframe - Complete frames of the play
        'players_df' - Dataframe - Contains player information, including weight
    Returns:
        'analysis_frames' - Dataframe - Compete frames of the play with pass_rusher force added
    '''
    # Get conversion rate from (pounds * yards) / seconds^2 to Newtons 
    force_conversion = 0.414764863 
    
    # Pull mass from players_df    
    mass = players_df[players_df.nflId == analysis_frames.pass_rusher.max()].weight.iat[0]

    # Calculate force = mass * acceleration * pursuit factor
    analysis_frames['pass_rusher_force_to_ball'] = round(mass * force_conversion * analysis_frames['pass_rusher_a'] * analysis_frames.pursuit_factor, 2)
    
    return analysis_frames


def create_true_pursuit(analysis_frames):
    '''
    Creates a metric that takes into account both the player's movement towards the ball, as well as the ball's
    movement away from the player, comparing those movements to get a truer pursuit factor.
    
    Parameters:
        'analysis_frames' - Dataframe - Complete frames of the play
    Returns:
        'analysis_frames' - Dataframe - Complete frames of the play with new metric added in
    '''
    true_pursuit_list = []
    
    for i in analysis_frames.index:
        true_pursuit_list.append(round((analysis_frames.pass_rusher_to_ball_vector_x.iloc[i]**2 +
                                        analysis_frames.pass_rusher_to_ball_vector_y.iloc[i]**2)**.5 -
                                       (analysis_frames.ball_to_pass_rusher_vector_x.iloc[i]**2 +
                                        analysis_frames.ball_to_pass_rusher_vector_y.iloc[i]**2)**.5,4))
                                 
    analysis_frames['pursuit_vs_escape'] = true_pursuit_list
    
    return analysis_frames


def create_pursuit1(analysis_frames):
    '''
    Creates a metric that compares the true pursuit to the change in distance between the pass rusher and the ball
    
    Parameters:
        'analysis_frames' - Dataframe - Complete frames of the play
    Returns:
        'analysis_frames' - Dataframe - Complete frames of the play with new metric added in
    '''
    pursuit_list = []
    
    for i in analysis_frames.index:
        frame_pursuit = round(analysis_frames.pursuit_vs_escape.iloc[i] /
                              analysis_frames.change_in_pass_rusher_to_ball_dist.iloc[i],4)
        
        # In order to account for a few manthematical anamolys that occur when the distance between player and ball 
        # stays constant, we set the pursuit metric to -8 or +6, around 4 std from mean.
        # Also, inverting negative to positive as it makes more sense to have positive numbers
        if frame_pursuit > 8:
            pursuit_list.append(-8)
            
        elif frame_pursuit < -6:
            pursuit_list.append(6)
            
        else:
            pursuit_list.append(frame_pursuit * -1)
        
    analysis_frames['pursuit1'] = pursuit_list
    
    return analysis_frames


def create_pursuit2(analysis_frames):
    '''
    Creates a metric that compares the true pursuit to the change in distance between the pass rusher and the ball
    
    Parameters:
        'analysis_frames' - Dataframe - Complete frames of the play
    Returns:
        'analysis_frames' - Dataframe - Complete frames of the play with new metric added in
    '''
    pursuit_list = []
    
    for i in analysis_frames.index:
        frame_pursuit = round(analysis_frames.pursuit_vs_escape.iloc[i] *
                              analysis_frames.pass_rusher_to_ball_dist_ratio.iloc[i],4)
        
        # In order to account for a few manthematical anamolys that occur when the distance between player and ball 
        # stays constant, we set the pursuit metric to -8 or +6, around 4 std from mean.
        # Also, inverting negative to positive as it makes more sense to have positive numbers
#         if frame_pursuit > 8:
#             pursuit_list.append(-8)
            
#         elif frame_pursuit < -6:
#             pursuit_list.append(6)
            
#         else:
        pursuit_list.append(frame_pursuit)
        
    analysis_frames['pursuit2'] = pursuit_list
    
    return analysis_frames


def create_pursuit3(analysis_frames):
    '''
    Throwing a Hail Mary...
    '''
    p3 = []
    
    for i in analysis_frames.index:
        ball_distance = (analysis_frames.ball_distance_moved_x.iloc[i]**2 +
                         analysis_frames.ball_distance_moved_y.iloc[i]**2)**.5
        player_towards_ball_dist = (analysis_frames.pass_rusher_to_ball_vector_x.iloc[i]**2 +
                                     analysis_frames.pass_rusher_to_ball_vector_y.iloc[i]**2)**.5
        
        if ball_distance == 0:
            p3.append(player_towards_ball_dist)
        else:
            p3.append(player_towards_ball_dist/ball_distance)
        
    analysis_frames['pursuit3'] = p3

        
    return analysis_frames


def create_pursuit4(analysis_frames):
    '''
    Throwing ANOTHER Hail Mary...
    '''
    p4 = []
    
    for i in analysis_frames.index:
        ball_distance = (analysis_frames.ball_distance_moved_x.iloc[i]**2 +
                         analysis_frames.ball_distance_moved_y.iloc[i]**2)**.5
        player_towards_ball_dist = analysis_frames.pursuit_vs_escape.iloc[i]
        
        if ball_distance == 0:
            p4.append(player_towards_ball_dist)
        else:
            p4.append(player_towards_ball_dist/ball_distance)
        
  
    analysis_frames['pursuit4'] = p4

        
    return analysis_frames


# ----- Support Functions -----------------------------------------------------------------------------

def reorientate_coord(origin, old):
    '''
    This takes in an old x or y coordinate and then remaps their values to an origin coordinate (0,0) which is the
    point_of_scrimmage (line of scrimmage and y location on width of field).
    
    Parameters:
        'origin' - Tuple of Floats - x or y coordinate of snap (point_of_scrimmage)
        'old' - Float - Old x or y coordiante to be reset to new point of scrimmage based origin
    Returns:
        'mod' - Float - New x or y coordinate based on a point of scrimmage origin.
    '''
    # Modified point
    mod = old - origin
    
    return mod


def euclidean_distance(x1, y1, x2, y2):
    '''
    Finds the straight line distance between two points in R2 (the football field).
    
    Parameters:
        'x1' - Float - X-coorindate of first point
        'y1' - Float - Y-coorindate of first point
        'x2' - Float - X-coorindate of second point
        'y2' - Float - Y-coorindate of second point
    Returns:
        distance - Float - The distance between the two points, in yards
    '''
    # Use the pythagorean theorem
    distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    
    return distance


def get_pass_blockers(analysis_frames):
    '''
    Since there can be anywhere from 1 to 4 pass_blockers (and possibly more in other datasets), this function 
    gets the prefixes such that metrics creation functions can be applied.
    
    Parameters:
        'analysis_frames' - Dataframe - Complete frames of the play ready for analysis
    Returns:
        'pass_blocker_list' - List of Strings - List of all pass_blocker column names
    '''    
    pass_blocker_list = []
    
    for col in analysis_frames.columns:
        if re.search(r'pass_blocker_?\d?$', col):
            pass_blocker_list.append(col)
            
    return pass_blocker_list


def pass_rusher_to_ball_colinearity_calc(pass_rusher_distance_moved_x,
                                         pass_rusher_distance_moved_y,
                                         ball_distance_moved_x,
                                         ball_distance_moved_y):
    '''
    Compares how directionally aligned the movement of the player and the movement of the ball are.
    
    Parameters:
        'pass_rusher_distance_moved_x' - Float - Pulled from analysis frames
        'pass_rusher_distance_moved_y' - Float - Pulled from analysis frames
        'ball_distance_moved_x' - Float - Pulled from analysis frames
        'ball_distance_moved_y' - Float - Pulled from analysis frames
    Returns:
        'mvmt_colinearity' - Float (-1 to 1) - How aligned the player's movement is with the ball movement
    '''
    # Colinearity goes from -1 to 1, with 1 being perfectly aligned, 0 being orthogonal, and -1 being opposite
    mvmt_colinearity = 1 - spatial.distance.cosine([pass_rusher_distance_moved_x,
                                                    pass_rusher_distance_moved_y], 
                                                   [ball_distance_moved_x,
                                                    ball_distance_moved_y])
                            
    return mvmt_colinearity


def pursuit_factor_calc(pass_rusher_to_ball_vector_x,
                        pass_rusher_to_ball_vector_y,
                        pass_rusher_distance_moved_x,
                        pass_rusher_distance_moved_y):
    '''
    Determines the pursuit angle (cosine distance) between the x and y components of the vector of the player 
    and their next location, and the player and the ball, using similarity (1 - angular distance)
    
    Parameters:
        'pass_rusher_to_ball_vector_x' - Float - Pulled from analysis_frames
        'pass_rusher_to_ball_vector_y' - Float - Pulled from analysis_frames
        'pass_rusher_distance_moved_x' - Float - Pulled from analysis_frames
        'pass_rusher_distance_moved_y' - Float - Pulled from analysis_frames 
    Returns:
        'pursuit_factor' - Float (-1 to 1) - How much of the player's movement is going towards the ball
    '''
    # Pursuit factor goes from -1 to 1, with 1 being perfectly aligned, 0 being orthogonal, and -1 being opposite
    pursuit_factor = 1 - spatial.distance.cosine([pass_rusher_to_ball_vector_x,
                                                  pass_rusher_to_ball_vector_y], 
                                                 [pass_rusher_distance_moved_x,
                                                  pass_rusher_distance_moved_y])
                            
    return pursuit_factor  


def component(v1x, v1y, on_v2x, on_v2y):
    '''
    Finds components of one vector [v1x, v1y] that lie along another vector [on_v2x, on_v2y].
    
    Parameters:
        'v1x' - Float - The x of the first vector.
        'v1y' - Float - The y of the first vector.
        'on_v2x' - Float - The x component of the vector we are finding the component of [v1x,v1y] on.
        'on_v2y' - Float - The y component of the vector we are finding the component of [v1x,v1y] on.
    Returns:
        'component' - Array of Floats - The vector representing the component of [v1x,v1y] on [on_v2x, on_v2y]
    '''
    v1 = [v1x, v1y]
    v2 = [on_v2x, on_v2y]
    
    dot_product = np.dot(v1, v2)
    magnitude = np.linalg.norm(v2)
    unit_vector = v2 / magnitude
    component = dot_product / magnitude * unit_vector
    
    return component


def escape_factor_calc(ball_distance_moved_x,
                       ball_distance_moved_y,
                       ball_to_pass_rusher_vector_x,
                       ball_to_pass_rusher_vector_y):
    '''
    Determines the pursuit angle (cosine distance) between the x and y components of the vector of the player 
    and their next location, and the player and the ball, using similarity (1 - angular distance)
    
    Parameters:
        'ball_distance_moved_x' - Float - Pulled from analysis_frames
        'ball_distance_moved_y' - Float - Pulled from analysis_frames
        'ball_to_pass_rusher_vector_x' - Float - Pulled from analysis_frames
        'ball_to_pass_rusher_vector_y' - Float - Pulled from analysis_frames 
    Returns:
        'escape_factor' - Float (-1 to 1) - How much of the ball's movement is going away from player
    '''
    # Pursuit factor goes from -1 to 1, with 1 being perfectly aligned, 0 being orthogonal, and -1 being opposite
    escape_factor = 1 - spatial.distance.cosine([ball_distance_moved_x,
                                                 ball_distance_moved_y], 
                                                [ball_to_pass_rusher_vector_x,
                                                 ball_to_pass_rusher_vector_y])
                            
    return escape_factor




# ====================================================================================================
# GRAPHING FUNCTIONS
# ====================================================================================================

def create_single_player_graph(analysis_frames, point_of_scrimmage):
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
    
    # Create ball and pass rusher variables to use
    index = analysis_frames.index
    b_x = analysis_frames.ball_x
    b_y = analysis_frames.ball_y
    pr_x = analysis_frames.pass_rusher_x
    pr_y = analysis_frames.pass_rusher_y
    pr_to_b_x = analysis_frames.pass_rusher_to_ball_vector_x
    pr_to_b_y = analysis_frames.pass_rusher_to_ball_vector_y
    
    # Create and graph the datapoints
    sns.scatterplot(x = b_x, y = b_y, hue = index, palette = 'Greens', legend = False)
              
    for mark in zip(analysis_frames.index,analysis_frames.ball_x,analysis_frames.ball_y):
        plt.annotate(mark[0],
                    (mark[1], mark[2]),
                    textcoords = 'offset points',
                    xytext = (4,4),
                    ha = 'center',
                    fontsize = 8)  
    
    # Create the frame labels for the football
    sns.scatterplot(x = pr_x, y = pr_y, hue = index, palette = 'Reds', legend = False)
    
    # Create the frame labels for the pass rusher
    for mark in zip(index,
                    pr_x,
                    pr_y,
                    pr_to_b_x,
                    pr_to_b_y):
        plt.annotate(mark[0],
                    (mark[1], mark[2]),
                    textcoords = 'offset points',
                    xytext = (4,4),
                    ha = 'center',
                    fontsize = 8)
        plt.annotate("",
                     xy=(mark[1],mark[2]), xycoords='data',
                     xytext=(mark[1] + mark[3], mark[2] + mark[4]), textcoords='data',
                     arrowprops=dict(arrowstyle="<-", connectionstyle="arc3"))
                    
    # Create pass blockers
    pass_blockers = get_pass_blockers(analysis_frames)
    
    for pass_blocker in pass_blockers:
        pb_x = analysis_frames[f'{pass_blocker}_x']
        pb_y = analysis_frames[f'{pass_blocker}_y']

        sns.scatterplot(x = pb_x, y = pb_y, hue = index, palette = 'Blues', legend = False)

        for mark in zip(index, pb_x, pb_y):
            plt.annotate(mark[0],
                        (mark[1], mark[2]),
                         textcoords = 'offset points',
                         xytext = (4,4),
                         ha = 'center',
                         fontsize = 8)
            
    # At figsize = [12,12], the line width of the line of scrimmage = 19.2; change in proportion to this
    plt.axvline(x = 0,
                lw = 14.4,
                alpha = 0.2)
    plt.axvline(x = 0)
    plt.text(0, (analysis_frames.ball_y.mean() + analysis_frames.pass_rusher_y.mean())/2, 
            f'Line of Scrimmage\n~{int(point_of_scrimmage[0])} yard line',
            fontsize = 15, rotation = 90,
            ha = 'center',
            va = 'top')

    plt.show()




# ====================================================================================================
# TESTING FUNCTIONS
# ====================================================================================================

def get_random_play_and_player():
    '''
    Generates all the parameters needed to select a random player from a random play, along with his opponents.
    This allows testing of the frame and metrics builders.
    
    Parameters:
        NONE
    Returns:
        'game' - Integer - Game number (unique for season)
        'play' - Integer - Unique for game
        'nflId' - Integer - Unique Id of player being analyzed
        'player_type' - String - The type of player bring analyzed (pass_rusher or pass_blocker)
    '''
    # Pull a random play from a random game
    game, play = nfl.random_play()

    # Get all players in the randomly selected play
    play_players = nfl.get_players_in_play(game, play)

    # Use only players who are blockers or pass rushers
    play_players = play_players[play_players.role.isin(['Pass Block','Pass Rush'])]
    
    # Randomly select a player and get their player type
    nflId = np.random.choice(list(play_players.nflId))
    player_type = get_player_type(play_players, nflId)    

    return game, play, nflId, player_type


# ----- Support Functions -----------------------------------------------------------------------------

def get_player_type(play_players, nflId):
    '''
    Gets the player type ('pass_rusher' or 'pass_blocker') given an nflId
    
    Parameters:
        'play_players' - Dataframe - A dataframe of play players and their roles and positions
        'nflId' - Integer - The unique id for the primary player being analyzed
    Returns:
        'player_type' - String - The type of player bring analyzed (pass_rusher or pass_blocker)
    '''
    # Get the player type
    player_type = play_players[play_players.nflId == nflId].role.iat[0]

    # Rename from the data to the convention used in these programs
    if player_type == 'Pass Block':
        player_type = 'pass_blocker'

    elif player_type == 'Pass Rush':
        player_type = 'pass_rusher'
    
    else:
        print('Error, Try Again')

    return player_type
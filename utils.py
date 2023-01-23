import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import pandas as pd

def get_saccades(df, l=5):
    """
    Identify saccades in eye tracking data. Based on the algorithm described in https://github.com/tmalsburg/saccades
    
    Parameters:
    df: Pandas DataFrame
        A DataFrame containing the eye tracking data.
        Required columns: 'LXpix', 'LYpix', 'Time'.
    l: float
        The value of lambda to use in the saccade detection algorithm.
        Default value: 5.
    
    Returns:
    Pandas DataFrame
        A copy of the input DataFrame with an additional column 'saccade'
        indicating whether each sample corresponds to a saccade.
    """
    # Check that the input data frame has the necessary columns
    if "LXpix" not in df.columns:
        raise ValueError("Input data frame needs column 'LXpix'.")
    if "LYpix" not in df.columns:
        raise ValueError("Input data frame needs column 'LYpix'.")
    if "Time" not in df.columns:
        raise ValueError("Input data frame needs column 'Time'.")
    
    # Compute vx as the difference between LXpix and the previous LXpix and divide by the sampling interval
    vx = (df['LXpix'] - df['LXpix'].shift(1)) / df['Time'].diff() # in pixels per second
    # Compute vy as the difference between LYpix and the previous LYpix and divide by the sampling interval
    vy = (df['LYpix'] - df['LYpix'].shift(1)) / df['Time'].diff() # in pixels per second

    # We don't want NAs, as they make our life difficult later
    # on.  Therefore, fill in missing values with 0
    vx = vx.fillna(0)
    vy = vy.fillna(0)

    # Calculate msdx and msdy (median standard deviation) as the median absolute deviation of vx and vy, respectively
    msdx = np.sqrt(np.median(vx**2) - np.median(vx)**2)
    msdy = np.sqrt(np.median(vy**2) - np.median(vy)**2)


    radiusx = msdx * l  # Compute radiusx as msdx multiplied by lambda
    radiusy = msdy * l  # Compute radiusy as msdy multiplied by lambda

    # Compute sacc as True where the squares of vx/radiusx and vy/radiusy are greater than 1
    sacc = (vx/radiusx)**2 + (vy/radiusy)**2 > 1 # in pixels per second
    
    # Add sacc as a new column to samples
    df['Saccade'] = sacc
    df['Speed in x coordinate'] = vx  # Add vx as a new column to samples
    df['Speed in y coordinate'] = vy  # Add vy as a new column to samples
    return df


def aggregate_fixations(df):
    """
    Mark fixations based on Saccade information.
    
    Parameters:
    df: Pandas DataFrame
        A DataFrame containing the eye tracking data.
        Required columns: 'Saccade'.
    
    Returns:
    Pandas DataFrame
        The input DataFrame with updated 'fixation.id' column.
    """
    # Check that the input data frame has the necessary columns
    if "Saccade" not in df.columns:
        raise ValueError("Input data frame needs column 'Saccade'.")

    # In saccade.events a 1 marks the start of a saccade and a -1 the start of a fixation
    saccade_events = np.sign(np.concatenate(([0], np.diff(df["Saccade"].values.astype(int)))))
    
    # Add a column to df with the saccade events
    df["Saccade.event"] = saccade_events
    
    # New fixations start either when a saccade ends:
    df["Fixation.id"] = np.cumsum((saccade_events==-1))

    return df

def get_subsequent_fixation_centers(df):
    """
    Calculates the subsequent fixation centers from the eye tracking data in the given DataFrame `df`.
    """
    fixations = df[df["Saccade"] == False]
    return np.array([[F["LXpix"].median(), F["LYpix"].median()] for i, F in fixations.groupby("Fixation.id") if len(F) > 0])


def add_statistical_significance_bars(groups, p_values, ax=None):
    '''Add statistical significance bars to a seaborn plot.'''

    # Add statistical significance bars to plot
    for i, pair in enumerate(combinations(groups, 2)):
        p = p_values[i]

        # Columns corresponding to the datasets of interest
        x1 = pair[0]
        x2 = pair[1]

        # Get index of x1 in groups array
        x1_index = np.where(groups == x1)[0]
        # Get index of x2 in groups array
        x2_index = np.where(groups == x2)[0]

        # Plot the bar
        # compute required height for the bar
        height = max(ax.get_ylim()) + 0.1 * max(ax.get_ylim())
        # height += height_offset # add some offset to the height
        edge_width = height/100
        ax.plot([x1_index, x1_index, x2_index, x2_index], [height - edge_width, height + edge_width, height + edge_width, height - edge_width], lw=1, c='k')
        
        # Significance level
        if p < 0.001:
            sig_symbol = '***'
        elif p < 0.01:
            sig_symbol = '**'
        elif p < 0.05:
            sig_symbol = '*'
        else:
            sig_symbol = 'ns'
       
        # Plot the significance symbol
        text_height = height + 0.05*max(ax.get_ylim())
        ax.text((x1_index + x2_index) / 2, text_height, sig_symbol, ha='center', va='bottom', c='k')
    
def reorganise_data(csv_image, verbose=False):
    task = []
    stimulus = []
    n_fixations = []
    fixation_durations = []
    n_saccades = []
    mean_saccade_distances = []
    mean_saccade_durations = []

    for i, (csv, image) in enumerate(csv_image.items()):

        # parse the name
        task_name = csv.split('/')[-1].split('_')[2]
        stimulus_name = csv.split('/')[-1].split('_')[3]

        df = pd.read_csv(csv)
        df = get_saccades(df, l=20)
        df = aggregate_fixations(df)

        # covert time to seconds
        df["Time"] = (df["Time"] - np.min( df["Time"])) / 1000

        # read image
        im = plt.imread(image)
        h, w, c = im.shape
        within_image_df = df[(df["LXpix"] >= 0) & (df["LXpix"] <= w) & (df["LYpix"] >= 0) & (df["LYpix"] <= h)]

        # calculate the distance between the starting and ending points of each saccade
        saccade_starts = df[df['Saccade.event'] == 1]
        # get the starting points of the saccades
        saccade_starts_coords = saccade_starts[['LXpix', 'LYpix']].values
        

        saccade_starts_time = saccade_starts['Time'].values
        # same for the ending points
        saccade_ends = df[df['Saccade.event'] == -1]
        # get the ending points of the saccades
        saccade_ends_coords = saccade_ends[['LXpix', 'LYpix']].values
        saccade_ends_time = saccade_ends['Time'].values

        if len(saccade_starts_coords) == len(saccade_ends_coords):
            # append to list
            task.append(task_name)
            stimulus.append(stimulus_name)
            # save the number of saccades   
            n_saccades.append(len(saccade_starts_coords))
            # calculate the distance between the starting and ending points of each saccade
            saccade_distances = np.linalg.norm(saccade_starts_coords - saccade_ends_coords, axis=1) 
            # remove nans
            saccade_distances = saccade_distances[~np.isnan(saccade_distances)]


            # calculate the duration of each saccade
            saccade_durations = saccade_ends_time - saccade_starts_time # pixels per second
            
            # append to list
            mean_saccade_durations.append(np.mean(saccade_durations))
            mean_saccade_distances.append(np.mean(saccade_distances))
            
            centers = get_subsequent_fixation_centers(within_image_df)
            n_fixations.append(len(centers))

            # compute duration of each fixation
            fixation_duration = []
            for i, F in within_image_df.groupby("Fixation.id"):
                if len(F) > 0:
                    fixation_duration.append(F["Time"].max() - F["Time"].min())
            fixation_durations.append(np.mean(fixation_duration))

            if verbose: 
                print(f"{csv} has {len(saccade_starts_coords)} saccade starts and {len(saccade_ends_coords)} saccade ends"
                      f" and {len(centers)} subsequent fixation"
                        f" and {len(fixation_duration)} fixation durations"
                        )

        else: 
            if verbose:
                print(f"Warning: {csv} has {len(saccade_starts_coords)} saccade starts and {len(saccade_ends_coords)} saccade ends")
    return task, stimulus, n_fixations, fixation_durations, n_saccades, mean_saccade_distances, mean_saccade_durations

def reorganise_data_with_time_intervals(csv_image, verbose=False):
    task = []
    stimulus = []
    n_fixations = []
    fixation_durations = []
    n_saccades = []
    mean_saccade_distances = []
    mean_saccade_durations = []
    time_interval = []

    for csv, image in csv_image.items():

        # parse the name
        task_name = csv.split('/')[-1].split('_')[2]
        stimulus_name = csv.split('/')[-1].split('_')[3]

        df = pd.read_csv(csv)
        df = get_saccades(df, l=20)
        df = aggregate_fixations(df)

        # covert time to seconds
        df["Time"] = (df["Time"] - np.min( df["Time"])) / 1000
        # read image
        im = plt.imread(image)
        h, w, c = im.shape
        within_image_df = df[(df["LXpix"] >= 0) & (df["LXpix"] <= w) & (df["LYpix"] >= 0) & (df["LYpix"] <= h)]
        
        # get the first and second time interval
        first_time_interval = within_image_df[within_image_df["Time"] <= 20] # first 20 seconds
        second_time_interval = within_image_df[within_image_df["Time"] > 20] # second 20 seconds
        
        # for each time interval, calculate the parameters
        for i, df in enumerate([first_time_interval, second_time_interval]):
            # calculate the distance between the starting and ending points of each saccade
            saccade_starts = df[df['Saccade.event'] == 1]
            # get the starting points of the saccades
            saccade_starts_coords = saccade_starts[['LXpix', 'LYpix']].values
            

            saccade_starts_time = saccade_starts['Time'].values
            # same for the ending points
            saccade_ends = df[df['Saccade.event'] == -1]
            # get the ending points of the saccades
            saccade_ends_coords = saccade_ends[['LXpix', 'LYpix']].values
            saccade_ends_time = saccade_ends['Time'].values

            if len(saccade_starts_coords) == len(saccade_ends_coords):
                time_interval.append(i)
                # append to list
                task.append(task_name)
                stimulus.append(stimulus_name)
                # save the number of saccades   
                n_saccades.append(len(saccade_starts_coords))
                # calculate the distance between the starting and ending points of each saccade
                saccade_distances = np.linalg.norm(saccade_starts_coords - saccade_ends_coords, axis=1) 
                # remove nans
                saccade_distances = saccade_distances[~np.isnan(saccade_distances)]


                # calculate the duration of each saccade
                saccade_durations = saccade_ends_time - saccade_starts_time # pixels per second
                
                # append to list
                mean_saccade_durations.append(np.mean(saccade_durations))
                mean_saccade_distances.append(np.mean(saccade_distances))
                
                centers = get_subsequent_fixation_centers(within_image_df)
                n_fixations.append(len(centers))

                # compute duration of each fixation
                fixation_duration = []
                for __, F in within_image_df.groupby("Fixation.id"):
                    if len(F) > 0:
                        fixation_duration.append(F["Time"].max() - F["Time"].min())
                fixation_durations.append(np.mean(fixation_duration))

                if verbose: 
                    print(f"{csv} has {len(saccade_starts_coords)} saccade starts and {len(saccade_ends_coords)} saccade ends"
                        f" and {len(centers)} subsequent fixation"
                            f" and {len(fixation_duration)} fixation durations"
                            )

            else: 
                if verbose:
                    print(f"Warning: {csv} has {len(saccade_starts_coords)} saccade starts and {len(saccade_ends_coords)} saccade ends")
    return task, stimulus, n_fixations, fixation_durations, n_saccades, mean_saccade_distances, mean_saccade_durations, time_interval
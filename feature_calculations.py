import pandas as pd
import numpy as np

def feature_calculation(csv_file_path):


    df = pd.read_csv(csv_file_path)


    # Assuming you have a dataframe named 'df' with columns 'X' and 'Y' representing the coordinates
    print("Calculating linear speed")
    df.sort_values(by=['animal', 'time_vid'], inplace=True)  # assuming there is a 'Time' column
    df['delta_X'] = df.groupby('animal')['x'].diff()
    df['delta_Y'] = df.groupby('animal')['y'].diff()
    df['Linear Speed'] = np.sqrt(df['delta_X']**2 + df['delta_Y']**2)

    time_step = df['time'].diff().mode()[0]
    for interval in [5, 10, 30,60, 120]:
        window_size = int(interval / time_step)
        df[f'Smoothed Speed {interval}s'] = df.groupby('animal')['Linear Speed'].rolling(window_size, min_periods=1).mean().reset_index(0, drop=True)
    
    print("Calculating angular velocity")
    # %%
    df.sort_values(by=['animal', 'time'], inplace=True)
    df['delta_X'] = df.groupby('animal')['x'].diff()
    df['delta_Y'] = df.groupby('animal')['y'].diff()
    df['angle'] = np.arctan2(df['delta_Y'], df['delta_X'])
    df['angular_velocity'] = df.groupby('animal')['angle'].diff() / df.groupby('animal')['time'].diff()

    time_step = df['time'].diff().mode()[0]
    for interval in [5, 10, 30, 60, 120]:
        df[f'Smoothed Angular Velocity {interval}s'] = df.groupby('animal')['angular_velocity'].rolling(int(interval / time_step), min_periods=1).mean().reset_index(0, drop=True)

    print("Calculating angular acceleration")
    df['angular_acceleration'] = df.groupby('animal')['angular_velocity'].diff() / df.groupby('animal')['time'].diff()

    for interval in [5, 10, 30, 60, 120]:
        df[f'Smoothed Angular Acceleration {interval}s'] = df.groupby('animal')['angular_acceleration'].rolling(int(interval / time_step), min_periods=1).mean().reset_index(0, drop=True)
    
    print("Calculating radial velocity")
    center_x, center_y = 540, 540
    df['distance'] = np.sqrt((df['x'] - center_x)**2 + (df['y'] - center_y)**2)
    df['radial_velocity'] = df.groupby('animal')['distance'].diff() / df.groupby('animal')['time'].diff()

    for interval in [5, 10, 30, 60, 120]:
        df[f'Smoothed Radial Velocity {interval}s'] = df.groupby('animal')['radial_velocity'].rolling(int(interval / time_step), min_periods=1).mean().reset_index(0, drop=True)

    print("Calculating radial acceleration")
    df['radial_acceleration'] = df.groupby('animal')['radial_velocity'].diff() / df.groupby('animal')['time'].diff()

    for interval in [5, 10, 30, 60, 120]:
        df[f'Smoothed Radial Acceleration {interval}s'] = df.groupby('animal')['radial_acceleration'].rolling(int(interval / time_step), min_periods=1).mean().reset_index(0, drop=True)

    print("Calculating tangential velocity")
    df['delta_x'] = df.groupby('animal')['x'].diff()
    df['delta_y'] = df.groupby('animal')['y'].diff()
    df['radial_vector_x'] = df['x'] - center_x
    df['radial_vector_y'] = df['y'] - center_y
    df['tangential_velocity'] = (df['radial_vector_x'] * df['delta_y'] - df['radial_vector_y'] * df['delta_x']) / df['distance']

    for interval in [5, 10, 30, 60, 120]:
        df[f'Smoothed Tangential Velocity {interval}s'] = df.groupby('animal')['tangential_velocity'].rolling(int(interval / time_step), min_periods=1).mean().reset_index(0, drop=True)

    print("Calculating tangential acceleration")
    df['tangential_acceleration'] = df.groupby('animal')['tangential_velocity'].diff() / df.groupby('animal')['time'].diff()

    for interval in [5, 10, 30, 60, 120]:
        df[f'Smoothed Tangential Acceleration {interval}s'] = df.groupby('animal')['tangential_acceleration'].rolling(int(interval / time_step), min_periods=1).mean().reset_index(0, drop=True)

    print("Calculating radial distance")
    # %%
    df['radial_distance'] = np.sqrt((df['x'] - center_x)**2 + (df['y'] - center_y)**2)

    for interval in [5, 10, 30, 60, 120]:
        df[f'Smoothed Radial Distance {interval}s'] = df.groupby('animal')['radial_distance'].rolling(int(interval / time_step), min_periods=1).mean().reset_index(0, drop=True)


    print("doing some cleaning")

    df = df[df['radial_distance'] <= 540]

    ####Dropping NaN values####
    # Set the threshold for NaN values
    threshold = 0.8  # Specify the threshold as a fraction (e.g., 0.5 for 50% NaN values)

    # Calculate the number of NaN values in each column
    nan_counts = df.isna().sum()

    # Get the columns that exceed the threshold
    columns_to_drop = nan_counts[nan_counts / len(df) > threshold].index

    # Drop the columns that exceed the threshold
    df = df.drop(columns=columns_to_drop)
    df = df.drop(columns=['time_vid'])

    counts = df['ground_truth'].value_counts()
    print(counts)

    return(df)

def counts(df):
    counts = df['ground_truth'].value_counts()
    print(counts)

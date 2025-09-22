#----------------- LOADING AND PREPROCESSING -----------------#

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from tslearn.clustering import TimeSeriesKMeans
from scipy.spatial.distance import pdist
from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model
from minisom import MiniSom
import pandas as pd
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import pandas as pd
import numpy as np
import os
from sklearn.metrics import adjusted_rand_score
import random
import datetime


def load_data(input_path):
    """
    Loads the data as DataFrame

    Parameters:
    - input_path (str): Path from where to load the data.

    Returns:
    - pandas.DataFrame
    """
    dataframes = []
    for filename in tqdm(os.listdir(input_path), desc='Loading data...'):
        if filename.endswith('.parquet'):
            # Assuming the building ID is the filename without the extension
            building_id = filename.replace('.parquet', '')
            filepath = os.path.join(input_path, filename)
            # Read the parquet file
            df = pd.read_parquet(filepath, engine='pyarrow')
            # Add a new column with the building ID
            df['building_id'] = int(building_id)
            # Append the dataframe to the list
            dataframes.append(df)
    print("Concatenating data...")   
    concatenated_dfs = pd.concat(dataframes, ignore_index=True)
    
    return concatenated_dfs

def normalize_seasonwise(data, seasons, scaler=MinMaxScaler()):
    """
    Normalize the input data separately for each season and each building specified by the seasons parameter.

    Parameters:
    - data (pandas.DataFrame): Input data to be normalized.
    - seasons (dict): Dictionary containing seasons as keys and corresponding months as values.
    - scaling (str): Define which scaler to use (e.g. 'MinMax' or 'MeanVariance'). 

    Returns:
    - pandas.DataFrame: Normalized data array or dataframe suitable for clustering or plotting.
    """
    normalized_dataframes = []

    for season, months in seasons.items():
        # Filter data for the current season
        season_data = data[data['timestamp'].dt.month.isin(months)]
        
        for building_id, group in season_data.groupby('building_id'):
            data_pivot = group.pivot(index='timestamp', columns='building_id', values='out.electricity.total.energy_consumption')
            data_array = data_pivot.values  # Convert to numpy array for clustering
                
            normalized_array = scaler.fit_transform(data_array)
            normalized_df = pd.DataFrame(normalized_array, index=data_pivot.index, columns=data_pivot.columns)
            normalized_df = normalized_df.stack().reset_index().rename(columns={0: 'out.electricity.total.energy_consumption'})
            normalized_df['season'] = season
            normalized_dataframes.append(normalized_df)

    # Concatenate normalized data for all seasons
    concatenated_normalized_dfs = pd.concat(normalized_dataframes, ignore_index=True)
    return concatenated_normalized_dfs
    
def normalize_across_dataset(data, scaler=MinMaxScaler()):
    """
    Normalize the input data per row.

    Parameters:
    - data (pandas.DataFrame): Input data to be normalized.
    - scaling (str): Define which scaler to use (e.g. 'MinMax' or 'MeanVariance'). 

    Returns:
    - pandas.DataFrame: Normalized data array or dataframe suitable for clustering or plotting.
    """
    normalized_dataframes = []
    for building_id, group in tqdm(data.groupby('building_id'), desc='Normalizing data for each building'):
        data_pivot = group.pivot(index='timestamp', columns='building_id', values='out.electricity.total.energy_consumption')
        data_array = data_pivot.values
        
        normalized_array = scaler.fit_transform(data_array)
        normalized_df = pd.DataFrame(normalized_array, index=data_pivot.index, columns=data_pivot.columns)
        normalized_df = normalized_df.stack().reset_index().rename(columns={0: 'out.electricity.total.energy_consumption'})
        normalized_dataframes.append(normalized_df)
    
    concatenated_normalized_dfs = pd.concat(normalized_dataframes, ignore_index=True)
    return concatenated_normalized_dfs

#----------------- EDA -----------------#
def check_peak_usage(df, threshold):
    """
    Check peak usage across buildings.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing the data for all buildings.
    - threshold (float): Threshold for peak usage.

    Returns:
    - peak_counts (list): List of peak counts for each building.
    - total_counts (list): List of total counts for each building.
    """
    peak_counts = []
    total_counts = []

    for building_id in df['building_id'].unique():
        building_df = df[df['building_id'] == building_id]
        peak_count = (building_df['out.electricity.total.energy_consumption'] > threshold).sum()
        total_count = building_df.shape[0]
        peak_counts.append(peak_count)
        total_counts.append(total_count)

    return peak_counts, total_counts

def calculate_statistics(peak_counts, total_counts):
    """
    Calculate statistics based on peak counts and total counts.

    Parameters:
    - peak_counts (list): List of peak counts for each building.
    - total_counts (list): List of total counts for each building.

    Returns:
    - customers_with_peaks (int): Number of customers with at least one peak.
    - total_customers (int): Total number of customers.
    - peak_occurrence_percentage (list): List of peak occurrence percentages for each building.
    """
    customers_with_peaks = sum(1 for count in peak_counts if count > 0)
    total_customers = len(total_counts)
    peak_occurrence_percentage = [(peak / total) * 100 for peak, total in zip(peak_counts, total_counts) if total > 0]

    return customers_with_peaks, total_customers, peak_occurrence_percentage

def aggregate_by_hour(df):
    df['hour'] = df['timestamp'].dt.hour
    hourly_data = df.groupby('hour')['out.electricity.total.energy_consumption'].mean().reset_index()
    return hourly_data

def extract_dst_period(data, dst_date, period_days=7):
    before_dst = data[(data['timestamp'] >= dst_date - pd.Timedelta(days=period_days)) &
                      (data['timestamp'] < dst_date)]
    after_dst = data[(data['timestamp'] >= dst_date) &
                     (data['timestamp'] < dst_date + pd.Timedelta(days=period_days))]
    return before_dst, after_dst


#----------------- METHODS -----------------#

def plot_cluster_distributions(final_labels):
    """
    Plot the distribution of data entries among different clusters.

    Parameters:
    - final_labels (array-like): Array containing cluster labels assigned to each data entry.

    Returns:
    - None (displays a bar plot).
    """
    unique, counts = np.unique(final_labels, return_counts=True)
    cluster_counts = dict(zip(unique, counts))

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.bar(cluster_counts.keys(), cluster_counts.values(), color='skyblue')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Entries')
    plt.title('Distribution of Entries Across Clusters')
    plt.xticks(list(cluster_counts.keys()))  # Ensure each cluster is labeled
    plt.show()
    
def plot_clusters(final_labels, data_normalized, duration='all', offset=0):
    """
    Plot time series clusters and their mean consumption.

    Parameters:
    - final_labels (array-like): Array containing cluster labels assigned to each data entry.
    - data_normalized (numpy.ndarray): Normalized data array suitable for clustering.
    - duration (str, optional): Duration for which to display data ('all', 'week', 'month', etc.). Defaults to 'all'.
    - offset (in hours) when plotting
    Returns:
    - None (displays plots).
    """
    # Calculate mean consumption for each cluster
    cluster_means = {}
    for label in np.unique(final_labels):
        cluster_data = data_normalized[final_labels == label]
        cluster_means[label] = cluster_data.mean(axis=0)  # Mean across rows/buildings

    # Number of clusters
    n_clusters = len(np.unique(final_labels))

    # Determine the number of data points to display based on duration
    if duration == 'all':
        num_data_points = data_normalized.shape[1] + offset
    elif duration == 'week':
        num_data_points = 7 * 24 * 4 + offset  # 7 days * 24 hours * 4 (15-minute intervals)
    elif duration == 'month':
        num_data_points = 30 * 24 * 4 + offset  # 30 days * 24 hours * 4 (15-minute intervals)
    else:
        raise ValueError("Invalid duration. Options are 'all', 'week', or 'month'.")

    # Generate time labels for x-axis
    start_time = datetime.datetime(2024, 1, 1, 0, 15)  # Starting time
    time_labels = [start_time + datetime.timedelta(minutes=i*15) for i in range(num_data_points)]

    # Create subplots - one for each cluster
    fig, axs = plt.subplots(n_clusters, 1, figsize=(15, n_clusters * 5), sharex=True)

    # Plot each building's time series and the cluster mean
    for ax, label in zip(axs, np.unique(final_labels)):
        # Plot each building's time series
        cluster_data = data_normalized[final_labels == label][:, offset:num_data_points]
        for idx in range(cluster_data.shape[0]):
            ax.plot(time_labels[offset:num_data_points], cluster_data[idx], 'k-', alpha=0.5, linewidth=0.5)

        # Plot the cluster's mean consumption
        ax.plot(time_labels[offset:num_data_points], cluster_means[label][offset:num_data_points], 'r-', linewidth=2)

        # Formatting the plot
        ax.set_title(f'Cluster {label + 1}')
        ax.set_ylabel('Power Consumption')
        ax.grid(True)

    # Set common labels
    fig.text(0.5, 0.04, 'Time', ha='center', va='center')
    fig.text(0.06, 0.5, 'Normalized Values', ha='center', va='center', rotation='vertical')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.show()
    
def dunn_index(X, labels):
    """
    Compute the Dunn Index for clustering evaluation.
    
    Args:
        X (numpy.ndarray): Data points, shape (n_samples, n_features).
        labels (numpy.ndarray): Cluster labels for each data point, shape (n_samples,).
        
    Returns:
        float: Dunn Index value.
    """
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)
    
    # Calculate intra-cluster distances
    intra_cluster_distances = []
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 1:
            intra_cluster_distances.append(np.max(pdist(cluster_points)))
    
    # Calculate inter-cluster distances
    inter_cluster_distances = pdist(X)
    
    # Compute Dunn Index
    min_inter_distance = np.min(inter_cluster_distances)
    max_intra_distance = np.max(intra_cluster_distances)
    
    # Handle division by zero
    if max_intra_distance == 0:
        dunn_index = float('inf')  # Assign infinity or any other suitable default value
    else:
        dunn_index = min_inter_distance / max_intra_distance
        
    return dunn_index

def hartigan_index(X, labels):
    """
    Compute the Hartigan Index for clustering evaluation.
    
    Args:
        X (numpy.ndarray): Data points, shape (n_samples, n_features).
        labels (numpy.ndarray): Cluster labels for each data point, shape (n_samples,).
        
    Returns:
        float: Hartigan Index value.
    """
    # Compute within-cluster dispersion
    within_cluster_dispersion = 0
    for label in np.unique(labels):
        cluster_points = X[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        within_cluster_dispersion += np.sum((cluster_points - centroid) ** 2)
    
    # Compute between-cluster dispersion
    centroid_all = np.mean(X, axis=0)
    between_cluster_dispersion = 0
    for label in np.unique(labels):
        cluster_points = X[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        between_cluster_dispersion += len(cluster_points) * np.sum((centroid - centroid_all) ** 2)
    
    # Calculate Hartigan index
    hartigan_index = within_cluster_dispersion / between_cluster_dispersion
    return hartigan_index

def compute_metrics(X, labels, print_metrics=True):
    """
    Compute various clustering evaluation metrics.
    
    Args:
        X (numpy.ndarray): Data points, shape (n_samples, n_features).
        labels (numpy.ndarray): Cluster labels for each data point, shape (n_samples,).
        print_metrics (bool, optional): Whether to print the computed metrics. Defaults to True.
        
    Returns:
        tuple: Silhouette index, Davies-Bouldin index, Dunn index, Hartigan index, and Calinski-Harabasz index.
    """
    silhouette_index = silhouette_score(X, labels) # maximum class spread/variance
    davies_bouldin_index = davies_bouldin_score(X, labels) # maximum interclass-intraclass distance ratio
    dunn_idx = dunn_index(X, labels) # minimum inter-cluster distance to maximum intra-cluster
    hartigan_idx = hartigan_index(X, labels) # interclass-intraclass distance ratio
    calinski_harabasz = calinski_harabasz_score(X, labels) # interclass-intraclass distance ratio
    
    if print_metrics:
        print("Silhouette index:", silhouette_index)
        print("Davies-Bouldin index:", davies_bouldin_index)
        print("Dunn index:", dunn_idx)
        print("Hartigan index:", hartigan_idx)
        print("Calinski-Harabasz index:", calinski_harabasz)
        
    return [silhouette_index, davies_bouldin_index, dunn_idx, hartigan_idx, calinski_harabasz]

#----------------- Clustering the Local Manifold of Autoencoded Embedding -----------------#

def autoencoder(dims, act='relu'):
    n_stacks = len(dims) - 1
    x = Input(shape=(dims[0],), name='input')
    h = x
    for i in range(n_stacks - 1):
        h = Dense(dims[i + 1], activation=act, name='encoder_%d' % i)(h)
    h = Dense(dims[-1], name='encoder_%d' % (n_stacks - 1))(h)
    for i in range(n_stacks - 1, 0, -1):
        h = Dense(dims[i], activation=act, name='decoder_%d' % i)(h)
    h = Dense(dims[0], name='decoder_0')(h)

    return Model(inputs=x, outputs=h)

# ---------- Feature Based Approach ----------# 

def extract_statistical_features(df):
    HOLIDAYS = [
    '2018-01-01', '2018-01-15', '2018-02-19', '2018-05-28',
    '2018-07-04', '2018-09-03', '2018-10-08', '2018-11-11',
    '2018-11-12', '2018-11-22', '2018-12-25'
     ]

    HOLIDAYS = [pd.Timestamp(date) for date in HOLIDAYS]

    # Define holidays
    holidays = [pd.Timestamp(date) for date in HOLIDAYS]

    features = {}
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Ensure that the data is numeric
    df['out.electricity.total.energy_consumption'] = pd.to_numeric(df['out.electricity.total.energy_consumption'], errors='coerce')

    # Yearly statistics
    features['mean'] = df['out.electricity.total.energy_consumption'].mean()
    features['std'] = df['out.electricity.total.energy_consumption'].std()

    # Representative daily load profile (24-hour average)
    hourly_profile = df.groupby(df.index.hour)['out.electricity.total.energy_consumption'].mean()
    features.update({f'daily_{i}': hourly_profile.iloc[i] for i in range(24)})

    # Monthly statistics
    monthly_stats = df.resample('M').mean()
    for month in range(1, 13):
        month_data = monthly_stats.loc[monthly_stats.index.month == month, 'out.electricity.total.energy_consumption']
        if not month_data.empty:
            features[f'month_{month}_mean'] = month_data.values[0]

    # Seasonal statistics
    SEASONS_MONTHS_DICT = {'cold': [1, 2, 12], 'hot': [6, 7, 8], 'mild': [3, 4, 5, 9, 10, 11]}
    for season, months in SEASONS_MONTHS_DICT.items():
        season_stats = df[df.index.month.isin(months)]
        features[f'{season}_mean'] = season_stats['out.electricity.total.energy_consumption'].mean()
        features[f'{season}_std'] = season_stats['out.electricity.total.energy_consumption'].std()
        features[f'{season}_min'] = season_stats['out.electricity.total.energy_consumption'].min()
        features[f'{season}_max'] = season_stats['out.electricity.total.energy_consumption'].max()

    # Weekday/Weekend statistics
    df['day_of_week'] = df.index.dayofweek
    features['weekday_mean'] = df[df['day_of_week'] < 5]['out.electricity.total.energy_consumption'].mean()
    features['weekend_mean'] = df[df['day_of_week'] >= 5]['out.electricity.total.energy_consumption'].mean()

    # Holiday/Non-Holiday statistics
    df['is_holiday'] = df.index.isin(HOLIDAYS)
    features['holiday_mean'] = df[df['is_holiday']]['out.electricity.total.energy_consumption'].mean()
    features['non_holiday_mean'] = df[~df['is_holiday']]['out.electricity.total.energy_consumption'].mean()

    # 24-hour representative load profile for each season
    for season, months in SEASONS_MONTHS_DICT.items():
        season_profile = df[df.index.month.isin(months)].groupby(df[df.index.month.isin(months)].index.hour)['out.electricity.total.energy_consumption'].mean()
        features.update({f'{season}_{i}': season_profile.iloc[i] for i in range(24)})

    # 24-hour representative load profile for weekdays and weekends
    weekday_profile = df[df['day_of_week'] < 5].groupby(df[df['day_of_week'] < 5].index.hour)['out.electricity.total.energy_consumption'].mean()
    weekend_profile = df[df['day_of_week'] >= 5].groupby(df[df['day_of_week'] >= 5].index.hour)['out.electricity.total.energy_consumption'].mean()
    features.update({f'weekday_{i}': weekday_profile.iloc[i] if i in weekday_profile.index else 0 for i in range(24)})
    features.update({f'weekend_{i}': weekend_profile.iloc[i] if i in weekend_profile.index else 0 for i in range(24)})

    # Time and date of highest consumption (as day of the year)
    max_consumption_time = df['out.electricity.total.energy_consumption'].idxmax()
    max_day_of_year = max_consumption_time.dayofyear
    features['max_consumption_day_of_year'] = max_day_of_year

    return features

def run_clustering(features_df, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features_df)
    labels = kmeans.labels_
    return labels    

#----------------- Evalution -----------------#

def compute_rand_index_randomly_sampled(data, initial_labels, map_size_x, map_size_y, n_iterations=100, fraction=0.8):
    """
    Compute the mean adjusted Rand index for a SOM-based clustering model using sampling.

    Args:
        data (numpy.ndarray): Time series data points, shape (n_samples, n_timesteps).
        initial_labels (numpy.ndarray): Initial cluster labels for each data point, shape (n_samples,).
        map_size_x (int): SOM map size along the x-axis.
        map_size_y (int): SOM map size along the y-axis.
        n_iterations (int, optional): Number of iterations to perform sampling. Defaults to 10.

    Returns:
        float: Mean adjusted Rand index over all iterations.
    """
    rand_indices = []
    num_clusters = len(np.unique(initial_labels))

    for _ in range(n_iterations):
        sampled_ids = sample_ids_by_cluster(initial_labels, num_clusters, fraction)
        train_set = data[sampled_ids]

        remaining_ids = np.setdiff1d(np.arange(data.shape[0]), sampled_ids)
        test_set = data[remaining_ids]

        # Create a new model and fit it to the train set
        input_len = train_set.shape[1]  
        new_model = MiniSom(map_size_x, map_size_y, input_len, sigma=0.3, learning_rate=0.1)
        new_model.random_weights_init(train_set)
        new_model.train_random(train_set, 1000)

        # Predict clusters on the test set
        predicted_bmu_test_set = np.array([new_model.winner(x) for x in test_set])
        # Get the clusters (grid points) of the SOM
        clusters = {}
        for i, (x, y) in enumerate(predicted_bmu_test_set):
            if (x, y) not in clusters:
                clusters[(x, y)] = []
            clusters[(x, y)].append(i)

        predicted_labels_test_set = np.zeros(test_set.shape[0], dtype=int)

        for cluster_index, (cluster, building_indices) in enumerate(clusters.items()):
            for building_index in building_indices:
                predicted_labels_test_set[building_index] = cluster_index

        # Compute the Rand index
        initial_labels_test_set = initial_labels[remaining_ids]

        rand_index = adjusted_rand_score(initial_labels_test_set, predicted_labels_test_set)
        rand_indices.append(rand_index)

    mean_rand_index = np.mean(rand_indices)
    return mean_rand_index

def sample_ids_by_cluster(labels, num_clusters, fraction=0.8):
    """
    Randomly sample data point IDs for each cluster.

    Args:
        labels (numpy.ndarray): Cluster labels for each data point, shape (n_samples,).
        num_clusters (int): Number of clusters.
        fraction (float, optional): Fraction of data points to sample from each cluster. Defaults to 0.8.

    Returns:
        numpy.ndarray: Sampled data point IDs.
    """
    sampled_ids = []
    for cluster in range(num_clusters):
        cluster_ids = np.where(labels == cluster)[0]
        sampled_size = int(fraction * len(cluster_ids))
        sampled_ids.extend(random.sample(list(cluster_ids), sampled_size))
    return np.array(sampled_ids)
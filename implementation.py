import pandas as pd
import os
from math import radians, sin, cos, asin, sqrt
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal

def find_file(base_path: str, filename: str) -> str:
    '''
    This function will recursively search for filename within the directory base_path and its 
    sub-directories. It will return the absolute path of the file if it finds it otherwise it'll return
    None
    '''
    files = os.listdir(base_path)
    for file in files:
        #if we get a match, return early
        if file == filename:
            return os.path.join(base_path, file)
        #if the file is a directory, then search the directory for filename
        if os.path.isdir(os.path.join(base_path, file)):
            result = find_file(os.path.join(base_path, file), filename)
            #this function returns None if nothing is found, so if result is not
            #None we have found the file and can return early
            if result != None:
                return result
    #implicit None return

def read_data(filename: str) -> pd.DataFrame:
    '''
    This function takes in a filename and recursively searches for the file name in this
    directory. If it finds the file, it returns a pandas DataFrame with the file's data in it,
    otherwise it returns None
    '''
    base_path = os.getcwd()
    file_path = find_file(base_path, filename)
    if file_path == None:
        return None
    
    df = pd.read_csv(file_path, delimiter='\t', header=0)
    #drop any rows where the gps time is NaN (can't use these)
    df = df.drop(df[df.gpstime == '?'].index)
    #due to the format of the gpstime float we have to convert to a float before converting
    #to an int
    df = df.astype({'gpstime': float, 'latitude': float, 'longitude': float})
    df = df.astype({'gpstime': int})
    #drop systime, we are only interested in gpstime
    df = df.drop(['systime'], axis=1)
    return df

def collect_data(sensor_name: str, car_name: str='seat') -> pd.DataFrame:
    '''
    This function takes in a sensor name (e.g., ID1) and a car name (e.g., fiat) and returns
    a DataFrame with the columns changed to sensor_name_column (e.g., ID1_gpstime)
    '''
    df = read_data(f'{sensor_name}_{car_name}-gps-parsed.txt')
    columns_to_change = [column for column in df.columns if column != 'gpstime']
    mapping = {column: f'{sensor_name}_{column}' for column in columns_to_change}
    df = df.rename(columns=mapping)
    return df

def merge_data(data_frames: 'list[pd.DataFrame] | tuple[pd.DataFrame]', on: str) -> pd.DataFrame:
    '''
    This function takes in a list of DataFrames and merges them on the on key
    '''
    if len(data_frames) < 2:
        raise ValueError('data_frames must be a list of dataframes with at least two items')
    df = data_frames[0]
    for i in range(1, len(data_frames)):
        other_df = data_frames[i]
        df = pd.merge(df, other_df, on=on, how='outer')
    return df

def earth_distance(lat_lon_1: 'tuple[float]', lat_lon_2: 'tuple[float]'):
    '''
    Calculates the distance in meters between two points on earth given
    a (latitude, longitude) pair
    '''
    lat_1 = radians(lat_lon_1[0])
    lon_1 = radians(lat_lon_1[1])
    lat_2 = radians(lat_lon_2[0])
    lon_2 = radians(lat_lon_2[1])

    #haversine formula
    delta_lon = lon_2 - lon_1
    delta_lat = lat_2 - lat_1
    a = sin(delta_lat / 2)**2 + cos(lat_1) * cos(lat_2) * sin(delta_lon / 2)**2

    c = 2 * asin(sqrt(a))

    #average radius of the earth in meters
    R = 6.371e6

    return c * R
    
def generate_dynamic_graph(car_data: pd.DataFrame, threshold: float=15.0) -> dict:
    '''
    Generates a dynamic graph from car data. This data should be collected and merged prior
    to being used as input. The algorithm adds a node for each car at each timestep if the 
    location is defined for the car at that timestep. It then calculates the distance between
    each pair of cars and if the distance is less than or equal to threshold, it creates an
    edge between the two cars
    '''
    snapshots = {
        'edge_indices': [],
        'features': [],
        'targets': [],
        'edge_weights': [],
    }
    #tqdm adds a progress bar
    for i in tqdm(range(0, len(car_data.index), 15)):
        #this is to make sure we don't add any nodes twice
        lat_lon_1 = (car_data['ID1_latitude'].iloc[i], car_data['ID1_longitude'].iloc[i])
        lat_lon_2 = (car_data['ID2_latitude'].iloc[i], car_data['ID2_longitude'].iloc[i])
        lat_lon_3 = (car_data['ID3_latitude'].iloc[i], car_data['ID3_longitude'].iloc[i])
        lat_lon_4 = (car_data['ID4_latitude'].iloc[i], car_data['ID4_longitude'].iloc[i])
        positions = [lat_lon_1, lat_lon_2, lat_lon_3, lat_lon_4]
        #node attributes
        features = np.array([
            [lat_lon_1],
            [lat_lon_2],
            [lat_lon_3],
            [lat_lon_4],
        ])
        edges = []
        edge_weights = []
        targets = []
        #check every pair of lat/lon
        for j in range(len(positions)-1):
            position = positions[j]
            for k in range(j+1, len(positions)):
                other_position = positions[k]
                #if the type of lat/lon is str, it is not defined for the first car
                #so we don't want to consider any pairs involving this car
                #(the second car will be considered again in the outer loop)
                if type(position[0]) == str or type(position[1]) == str:
                    break
                #if the type of other lat/lon is a str, it is not defined, and we
                #need to skip over that position
                if type(other_position[0]) == str or type(other_position[1]) == str:
                    continue
                #get the earth_distance (using the haversine formula) between the two positions
                distance = earth_distance(position, other_position)
                #if the distance between the two cars is less than or equal to our threshold,
                #add an edge between the two nodes associated with those positions
                if distance <= threshold:
                    edges.append([j, k])
                    edges.append([k, j])
                    edge_weights.append(1)
                    edge_weights.append(1)
                targets.append(None)
        edge_index = np.array(edges).T
        edge_weights = np.array(edge_weights)
        targets = np.array(targets)
        snapshots['edge_indices'].append(edge_index)
        snapshots['features'].append(features)
        snapshots['edge_weights'].append(edge_weights)
        snapshots['targets'].append(targets)
    return snapshots

id_1 = collect_data('ID1')
id_2 = collect_data('ID2', 'audi')
id_3 = collect_data('ID3', 'nissan')
id_4 = collect_data('ID4', 'fiat')

df = merge_data((id_1, id_2, id_3, id_4), on='gpstime')

graph = generate_dynamic_graph(df)

dataset = DynamicGraphTemporalSignal(
    edge_indices=graph['edge_indices'],
    features=graph['features'],
    edge_weights=graph['edge_weights'],
    targets=graph['targets']
)

class GAE(torch.nn.Module):
    def __init__(self, node_features):
        super(GAE, self).__init__()
        self.recurrent = DCRNN(node_features, )
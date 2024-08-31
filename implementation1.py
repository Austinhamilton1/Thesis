import pandas as pd
import os
from math import radians, sin, cos, asin, sqrt, isnan
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric.nn.models import GAE
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
from torch_geometric_temporal import DCRNN

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

def read_gps_data(filename: str, direction: str) -> pd.DataFrame:
    '''
    This function takes in a filename and recursively searches for the file name in this
    directory. If it finds the file, it returns a pandas DataFrame with the file's data in it,
    otherwise it returns None
    '''
    base_path = os.path.join(os.getcwd(), 'GPS Data')
    base_path = os.path.join(base_path, f'{direction}_veh_files')
    file_path = find_file(base_path, filename)
    if file_path == None:
        return None
    
    df = pd.read_csv(file_path, delimiter=',', header=0)
    df = df.astype({
        'unixtime': int,
        'latitude': float,
        'longitude': float,
        'postmile': float,
        'speed': float,
    })

    return df

def collect_gps_data(vehicle_num: int, direction: str) -> pd.DataFrame:
    '''
    This function takes in a vehicle number (e.g., 175) and returns
    a DataFrame with the columns changed to column_number (e.g., latitude_175)
    '''
    df = read_gps_data(filename=f'veh_{vehicle_num}.csv', direction=direction)
    columns_to_change = [column for column in df.columns if column != 'unixtime']
    mapping = {column: f'{column}_{vehicle_num}' for column in columns_to_change}
    df = df.rename(columns=mapping)
    return df 


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
    df = df.drop(df[df.latitude == '?'].index)
    df = df.drop(df[df.longitude == '?'].index)
    df = df.drop(df[df.altitude == '?'].index)
    df = df.drop(df[df.speed == '?'].index)
    df = df.drop(df[df.heading == '?'].index)
    #due to the format of the gpstime float we have to convert to a float before converting
    #to an int
    df = df.astype({
        'gpstime': float,
        'latitude': float, 
        'longitude': float, 
        'altitude': float,
        'speed': float,
        'heading': float,
    })
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
    
def generate_dynamic_graph(car_data: pd.DataFrame, threshold: float=15.0) -> 'list[Data]':
    '''
    Generates a dynamic graph from car data. This data should be collected and merged prior
    to being used as input. The algorithm adds a node for each car at each timestep if the 
    location is defined for the car at that timestep. It then calculates the distance between
    each pair of cars and if the distance is less than or equal to threshold, it creates an
    edge between the two cars
    '''
    snapshots = []
    #tqdm adds a progress bar
    for i in tqdm(range(0, len(car_data.index))):
        northbound_features = [(car_data[f'latitude_{j+1}_nb'].iloc[i], car_data[f'longitude_{j+1}_nb'].iloc[i], car_data[f'postmile_{j+1}_nb'].iloc[i], car_data[f'speed_{j+1}_nb'].iloc[i]) for j in range(1387)]
        southbound_features = [(car_data[f'latitude_{j+1}_sb'].iloc[i], car_data[f'longitude_{j+1}_sb'].iloc[i], car_data[f'postmile_{j+1}_sb'].iloc[i], car_data[f'speed_{j+1}_sb'].iloc[i]) for j in range(1511)]
        features = [feature for feature in northbound_features]
        for feature in southbound_features:
            features.append(feature)
        
        positions = [(feature[0], feature[1]) for feature in features]
        
        #node attributes
        features = torch.tensor(features, dtype=torch.float)
        edges = []
        #check every pair of lat/lon
        for j in range(len(positions)-1):
            position = positions[j]
            for k in range(j+1, len(positions)):
                other_position = positions[k]
                #if the lat/lon is 0, it is not defined for the first car
                #so we don't want to consider any pairs involving this car
                #(the second car will be considered again in the outer loop)
                if position[0] == 0 or position[1] == 0:
                    break
                #if the other lat/lon is 0, it is not defined, and we
                #need to skip over that position
                if other_position[0] == 0 or other_position[1] == 0:
                    continue
                #get the earth_distance (using the haversine formula) between the two positions
                distance = earth_distance(position, other_position)
                #if the distance between the two cars is less than or equal to our threshold,
                #add an edge between the two nodes associated with those positions
                if distance <= threshold:
                    edges.append([j, k])
                    edges.append([k, j])
        #if the graph snapshot has no edges, then skip this snapshot
        if len(edges) == 0:
            continue
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        snapshot = Data(x=features, edge_index=edge_index)
        snapshots.append(snapshot)
    return snapshots

df = pd.read_csv('condensed_dataset.csv', header=0)

graph = generate_dynamic_graph(df, threshold=30.0)

transform = RandomLinkSplit(is_undirected=True)
dataset = []
for snapshot in graph:
    train, val, test = transform(snapshot)
    dataset.append({
        'train': train,
        'val': val,
        'test': test
    })

class GraphEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphEncoder, self).__init__()
        self.encoder1 = DCRNN(in_channels=in_channels, out_channels=2*out_channels, K=1)
        self.encoder2 = DCRNN(in_channels=2*out_channels, out_channels=out_channels, K=1)
    def forward(self, x, edge_index):
        x = self.encoder1(x, edge_index)
        x = F.relu(x)
        x = self.encoder2(x, edge_index)
        return x
    
out_channels = 2
num_features = dataset[0]['train'].num_features
epochs = 100

model = GAE(GraphEncoder(num_features, out_channels))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    for snapshot in dataset:
        x = snapshot['train'].x.to(device)
        train_pos_edge_index = snapshot['train'].edge_index.to(device)
        z = model.encode(x, train_pos_edge_index)
        train_neg_edge_index = negative_sampling(train_pos_edge_index, z.size(0)).long().to(device)
        loss = model.recon_loss(z, train_pos_edge_index, train_neg_edge_index)
        loss.backward()
        optimizer.step()

def test(pos_edge_indices, neg_edge_indices):
    model.eval()
    total_auc = 0.0
    total_ap = 0.0
    for i, snapshot in enumerate(dataset):
        with torch.no_grad():
            x = snapshot['train'].x.to(device)
            train_pos_edge_index = snapshot['train'].edge_index.to(device)
            z = model.encode(x, train_pos_edge_index)
        auc, ap = model.test(z, pos_edge_indices[i], neg_edge_indices[i])
        total_auc += auc
        total_ap += ap
    return total_auc / len(dataset), total_ap / len(dataset)

for epoch in range(1, epochs+1):
    train()
    pos_edge_indices = [data['test'].edge_index for data in dataset]
    neg_edge_indices = [negative_sampling(data['test'].edge_index) for data in dataset]
    auc, ap = test(pos_edge_indices, neg_edge_indices)
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
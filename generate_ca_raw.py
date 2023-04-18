import os.path as osp
import pandas as pd
import json
import shapely
from utils import get_root_dir

data_path = osp.join(get_root_dir(), 'data', 'ca', 'raw')
raw_checkins = pd.read_csv(osp.join(data_path, 'loc-gowalla_totalCheckins.txt'), sep='\t', header=None)
raw_checkins.columns = ['userid', 'datetime', 'checkins_lat', 'checkins_lng', 'id']
subset1 = pd.read_csv(osp.join(data_path, 'gowalla_spots_subset1.csv'))
raw_checkins_subset1 = raw_checkins.merge(subset1, on='id')

with open(osp.join(data_path, 'us_state_polygon_json.json'), 'r') as f:
    us_state_polygon = json.load(f)

for i in us_state_polygon['features']:
    if i['properties']['name'].lower() == 'california':
        california = shapely.polygons(i['geometry']['coordinates'][0])
    if i['properties']['name'].lower() == 'nevada':
        nevada = shapely.polygons(i['geometry']['coordinates'][0])

# check if the checkin took place in California or Nevada
raw_checkins_subset1['is_ca'] = raw_checkins_subset1.apply(
    lambda x: nevada.intersects(
        shapely.geometry.Point(x['checkins_lng'], x['checkins_lat'])) or california.intersects(
        shapely.geometry.Point(x['checkins_lng'], x['checkins_lat'])), axis=1
)
raw_checkins_subset1 = raw_checkins_subset1[raw_checkins_subset1['is_ca']]

df = raw_checkins_subset1[['userid', 'id', 'spot_categories', 'checkins_lat', 'checkins_lng', 'datetime']]
df.columns = ['UserId', 'PoiId', 'PoiCategoryId', 'Latitude', 'Longitude', 'UTCTime']
df.to_csv(osp.join(data_path, 'dataset_gowalla_ca_ne.csv'), index=False)

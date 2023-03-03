import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import pickle
import os.path as osp
from conf_parser import Cfg
from generate_hypergraph import generate_hypergraph_from_file


if __name__ == "__main__":
    root_path = './data/nyc_getnext'
    df_train = pd.read_csv(osp.join(root_path, 'raw', 'NYC_train.csv'))
    df_val = pd.read_csv(osp.join(root_path, 'raw', 'NYC_val.csv'))
    df_test = pd.read_csv(osp.join(root_path, 'raw', 'NYC_test.csv'))
    df_train['SplitTag'] = 'train'
    df_val['SplitTag'] = 'validation'
    df_test['SplitTag'] = 'test'
    df = pd.concat([df_train, df_val, df_test])
    df.columns = [
        'UserId', 'PoiId', 'PoiCategoryId', 'PoiCategoryCode', 'PoiCategoryName', 'Latitude', 'Longitude',
        'TimezoneOffset', 'UTCTime', 'UTCTimeOffset', 'UTCTimeOffsetWeekday', 'UTCTimeOffsetNormInDayTime',
        'pseudo_session_trajectory_id', 'UTCTimeOffsetNormDayShift', 'UTCTimeOffsetNormRelativeTime', 'SplitTag'
    ]
    df['trajectory_id'] = df['pseudo_session_trajectory_id']
    df['UTCTimeOffset'] = df['UTCTimeOffset'].apply(
        lambda x: datetime.strptime(x[:19], "%Y-%m-%d %H:%M:%S"))
    df['UTCTimeOffsetEpoch'] = df['UTCTimeOffset'].apply(lambda x: x.strftime('%s'))
    df['UTCTimeOffsetWeekday'] = df['UTCTimeOffset'].apply(lambda x: x.weekday())
    df['UTCTimeOffsetHour'] = df['UTCTimeOffset'].apply(lambda x: x.hour)
    df['UTCTimeOffsetDay'] = df['UTCTimeOffset'].apply(lambda x: x.strftime('%Y-%m-%d'))
    df['UserRank'] = df.groupby('UserId')['UTCTimeOffset'].rank(method='first')
    df = df.sort_values(by=['UserId', 'UTCTimeOffset'], ascending=True)
    df['check_ins_id'] = df['UTCTimeOffset'].rank(ascending=True, method='first') - 1

    trajectory_id_lb = LabelEncoder()
    trajectory_id_lb = trajectory_id_lb.fit(df['pseudo_session_trajectory_id'].values.tolist())
    df['pseudo_session_trajectory_id'] = [
        trajectory_id_lb.transform([i])[0] for i in df['pseudo_session_trajectory_id'].values.tolist()
    ]

    df['pseudo_session_trajectory_rank'] = df.groupby(
        'pseudo_session_trajectory_id')['UTCTimeOffset'].rank(method='first')
    df['query_pseudo_session_trajectory_id'] = df['pseudo_session_trajectory_id'].shift()
    df.loc[df['pseudo_session_trajectory_rank'] == 1, 'query_pseudo_session_trajectory_id'] = None
    df['last_checkin_epoch_time'] = df['UTCTimeOffsetEpoch'].shift()
    df.loc[df['pseudo_session_trajectory_rank'] == 1, 'last_checkin_epoch_time'] = None

    poi_id_lb = LabelEncoder()
    poi_id_lb = poi_id_lb.fit(df[df['SplitTag'] == 'train']['PoiId'].values.tolist())
    padding_poi_id = len(poi_id_lb.classes_)
    df['PoiId'] = [
        poi_id_lb.transform([i])[0] if i in poi_id_lb.classes_ else padding_poi_id
        for i in df['PoiId'].values.tolist()
    ]
    poi_category_lb = LabelEncoder()
    poi_category_lb = poi_category_lb.fit(df[df['SplitTag'] == 'train']['PoiCategoryId'].values.tolist())
    padding_poi_category = len(poi_category_lb.classes_)
    df['PoiCategoryId'] = [
        poi_category_lb.transform([i])[0] if i in poi_category_lb.classes_ else padding_poi_category
        for i in df['PoiCategoryId'].values.tolist()
    ]
    user_id_lb = LabelEncoder()
    user_id_lb = user_id_lb.fit(df[df['SplitTag'] == 'train']['UserId'].values.tolist())
    padding_user_id = len(user_id_lb.classes_)
    df['UserId'] = [
        user_id_lb.transform([i])[0] if i in user_id_lb.classes_ else padding_user_id
        for i in df['UserId'].values.tolist()
    ]

    hour_id_lb = LabelEncoder()
    hour_id_lb = hour_id_lb.fit(df[df['SplitTag'] == 'train']['UTCTimeOffsetHour'].values.tolist())
    padding_hour = len(hour_id_lb.classes_)
    df['UTCTimeOffsetHour'] = [
        hour_id_lb.transform([i])[0] if i in hour_id_lb.classes_ else padding_hour
        for i in df['UTCTimeOffsetHour'].values.tolist()
    ]

    weekday_id_lb = LabelEncoder()
    weekday_id_lb = weekday_id_lb.fit(df[df['SplitTag'] == 'train']['UTCTimeOffsetWeekday'].values.tolist())
    padding_weekday = len(weekday_id_lb.classes_)
    df['UTCTimeOffsetWeekday'] = [
        weekday_id_lb.transform([i])[0] if i in weekday_id_lb.classes_ else padding_weekday
        for i in df['UTCTimeOffsetWeekday'].values.tolist()
    ]

    # df['UserId'] = user_id_lb.transform(df['UserId'].values.tolist())
    with open(osp.join(root_path, 'processed', 'label_encoding.pkl'), 'wb') as f:
        pickle.dump([
            poi_id_lb, poi_category_lb, user_id_lb, hour_id_lb, weekday_id_lb,
            padding_poi_id, padding_poi_category, padding_user_id, padding_hour, padding_weekday
        ], f)

    df.loc[df['UserRank'] == 1, 'SplitTag'] = 'ignore'
    df.loc[df['pseudo_session_trajectory_rank'] == 1, 'SplitTag'] = 'ignore'
    df['pseudo_session_trajectory_count'] = df.groupby(
        'pseudo_session_trajectory_id')['UTCTimeOffset'].transform('count')
    df.loc[(df['SplitTag'] == 'validation') & (
            df['pseudo_session_trajectory_count'] != df['pseudo_session_trajectory_rank']
    ), 'SplitTag'] = 'ignore'
    df.loc[(df['SplitTag'] == 'test') & (
            df['pseudo_session_trajectory_count'] != df['pseudo_session_trajectory_rank']
    ), 'SplitTag'] = 'ignore'

    df['pseudo_aoi_trajectory_id'] = 0
    cols = ['check_ins_id', 'UTCTimeOffset', 'UTCTimeOffsetEpoch', 'pseudo_session_trajectory_id',
            'query_pseudo_session_trajectory_id', 'UserId', 'Latitude', 'Longitude', 'PoiId', 'PoiCategoryId',
            'PoiCategoryName', 'last_checkin_epoch_time', 'pseudo_aoi_trajectory_id', 'trajectory_id']
    print(
        f"validation shape: {df[df['SplitTag'] == 'validation'].shape}",
        f"test shape: {df[df['SplitTag'] == 'test'].shape}")
    df[df['SplitTag'] == 'train'][cols].to_csv(
        osp.join(root_path, 'processed', 'train_sample.csv'), index=False)
    df[df['SplitTag'] == 'validation'][cols].to_csv(
        osp.join(root_path, 'processed', 'validation_sample.csv'), index=False)
    df[df['SplitTag'] == 'test'][cols].to_csv(
        osp.join(root_path, 'processed', 'test_sample.csv'), index=False)
    df.to_csv(osp.join(root_path, 'processed', 'sample.txt'), index=False)

    cfg = Cfg("run.yml")
    data_dir = osp.join('data', 'nyc_getnext')
    processed_path = osp.join(data_dir, 'processed')
    processed_file = osp.join(processed_path, 'sample.txt')
    generate_hypergraph_from_file(processed_file, processed_path, cfg.dataset_args)

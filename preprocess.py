"""
@author: yifeng
@date: 2022-10-13
@description: preprocess for next poi
"""
import os
import os.path as osp
import pandas as pd
from datetime import datetime
from datetime import timedelta
from utils import haversine
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import pickle


class FileReaderBase:
    root_path = os.getcwd()

    @classmethod
    def read_dataset(cls, file_name, dataset_name):
        raise NotImplementedError


class TsmcFileReader(FileReaderBase):
    @classmethod
    def read_dataset(cls, file_name, dataset_name):
        file_path = osp.join(cls.root_path, 'raw', file_name)
        if dataset_name == 'gowalla':
            df = pd.read_csv(file_path, sep=',')
            df['UTCTimeOffset'] = df['UTCTime'].apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ"))
            df['PoiCategoryName'] = df['PoiCategoryId']
        else:
            df = pd.read_csv(file_path, sep='\t', encoding='latin-1', header=None)
            df.columns = [
                'UserId', 'PoiId', 'PoiCategoryId', 'PoiCategoryName', 'Latitude', 'Longitude', 'TimezoneOffset',
                'UTCTime'
            ]
            df['UTCTime'] = df['UTCTime'].apply(lambda x: datetime.strptime(x, "%a %b %d %H:%M:%S +0000 %Y"))
            df['UTCTimeOffset'] = df['UTCTime'] + df['TimezoneOffset'].apply(lambda x: timedelta(hours=x/60))
        df['UTCTimeOffsetEpoch'] = df['UTCTimeOffset'].apply(lambda x: x.strftime('%s'))
        df['UTCTimeOffsetWeekday'] = df['UTCTimeOffset'].apply(lambda x: x.weekday())
        df['UTCTimeOffsetHour'] = df['UTCTimeOffset'].apply(lambda x: x.hour)
        df['UTCTimeOffsetDay'] = df['UTCTimeOffset'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df['UserRank'] = df.groupby('UserId')['UTCTimeOffset'].rank(method='first')

        print(max(df['UTCTimeOffset']), min(df['UTCTimeOffset']))
        print(df.shape, df['UserId'].nunique(), df['PoiId'].nunique())
        return df

    @classmethod
    def do_filter(cls, df, poi_min_freq, user_min_freq):
        # print('-'*10, user_count[user_count['UserId'].isin((122, 410, 455, 668, 693, 942, 1073))])

        # poi_count = df.groupby('PoiId')['UserId'].count().reset_index()
        # user_count = df.groupby('UserId')['PoiId'].count().reset_index()
        # df = df[
        #     (df['PoiId'].isin(poi_count[poi_count['UserId'] > poi_min_freq]['PoiId'])) &
        #     (df['UserId'].isin(user_count[user_count['PoiId'] > user_min_freq]['UserId']))]

        poi_count = df.groupby('PoiId')['UserId'].count().reset_index()
        df = df[df['PoiId'].isin(poi_count[poi_count['UserId'] > poi_min_freq]['PoiId'])]
        user_count = df.groupby('UserId')['PoiId'].count().reset_index()
        df = df[df['UserId'].isin(user_count[user_count['PoiId'] > user_min_freq]['UserId'])]

        print(
            f"User count: {len(user_count)}, "
            f"Low frequency user count: {len(user_count[user_count['PoiId'] <= user_min_freq])}, "
            f"ratio: {len(user_count[user_count['PoiId'] <= user_min_freq]) / len(user_count):.5f}"
        )
        print(
            f"Poi count: {len(poi_count)}, "
            f"Low frequency poi count: {len(poi_count[poi_count['UserId'] <= poi_min_freq])}, "
            f"ratio: {len(poi_count[poi_count['UserId'] <= poi_min_freq]) / len(poi_count):.5f}"
        )

        # 按照用户和日期统计, 如果一天只有一个行为, 则去除
        # tmp = df.groupby(['UserId', 'UTCTimeOffsetDay'])['PoiId'].count().reset_index()
        # tmp = tmp[tmp['PoiId'] > 1]
        # df = df.merge(tmp[['UserId', 'UTCTimeOffsetDay']], on=['UserId', 'UTCTimeOffsetDay'])
        return df

    @classmethod
    def split_train_test(cls, df, mode, is_sorted=False):
        if not is_sorted:
            df = df.sort_values(by=['UserId', 'UTCTimeOffset'], ascending=True)
        # train_list, val_list, test_list = [], [], []
        if mode == 'global_time':
            max_time = max(df['UTCTimeOffset'])
            min_time = min(df['UTCTimeOffset'])
            time_diff = (max_time - min_time).days
            train_max_day = min_time + timedelta(days=int(time_diff * 0.8))
            val_max_day = min_time + timedelta(days=int(time_diff * 0.9))
            df['SplitTag'] = 'train'
            df.loc[
                (df['UTCTimeOffset'] >= train_max_day) & (df['UTCTimeOffset'] < val_max_day), 'SplitTag'] = 'validation'
            df.loc[df['UTCTimeOffset'] >= val_max_day, 'SplitTag'] = 'test'
            df.loc[df['UserRank'] == 1, 'SplitTag'] = 'ignore'
        if mode == 'global_count':
            df['UserRank'] = df.groupby('UserId')['UTCTimeOffset'].rank(method='first')
            df['SplitTag'] = 'train'
            total_len = df.shape[0]
            validation_index = int(total_len * 0.8)
            test_index = int(total_len * 0.9)
            print(f"validation index: {validation_index}", f"test index: {test_index}")
            df = df.sort_values(by='UTCTimeOffset', ascending=True)
            df.iloc[validation_index:test_index]['SplitTag'] = 'validation'
            df.iloc[test_index:]['SplitTag'] = 'test'
            # df.loc[df['UserRank'] == 1, 'SplitTag'] = 'ignore'

        # 去除train里面没有出现POI和user
        # train_user_set = set(df[df['SplitTag'] == 'train']['UserId'])
        # train_poi_set = set(df[df['SplitTag'] == 'train']['PoiId'])
        # df_validation = df[df['SplitTag'] == 'validation']
        # df_test = df[df['SplitTag'] == 'test']
        # df_validation = df_validation[
        #     (df_validation['UserId'].isin(train_user_set)) & (df_validation['PoiId'].isin(train_poi_set))]
        # df_test = df_test[(df_test['UserId'].isin(train_user_set)) & (df_test['PoiId'].isin(train_poi_set))]
        # df = pd.concat([df[df['SplitTag'].isin(('train', 'ignore'))], df_validation, df_test])
        # df.loc[df['UserRank'] == 1, 'SplitTag'] = 'ignore'

        df['UserRank'] = df.groupby('UserId')['UTCTimeOffset'].rank(method='first')

        # 过滤前后时间间隔超过24h的check-in
        df = df.sort_values(by=['UserId', 'UTCTimeOffset'], ascending=True)
        isolated_index = []
        for idx, diff1, diff2, user, user1, user2 in zip(
            df.index,
            df['UTCTimeOffset'].diff(1),
            df['UTCTimeOffset'].diff(-1),
            df['UserId'],
            df['UserId'].shift(1),
            df['UserId'].shift(-1)
        ):
            if pd.isna(diff1) and abs(diff2.total_seconds()) > 86400 and user == user2:
                isolated_index.append(idx)
            elif pd.isna(diff2) and abs(diff1.total_seconds()) > 86400 and user == user1:
                isolated_index.append(idx)
            if abs(diff1.total_seconds()) > 86400 and abs(diff2.total_seconds()) > 86400 and user == user1 and user == user2:
                isolated_index.append(idx)
            elif abs(diff2.total_seconds()) > 86400 and user == user2 and user != user1:
                isolated_index.append(idx)
            elif abs(diff1.total_seconds()) > 86400 and user == user1 and user != user2:
                isolated_index.append(idx)
        df = df[~df.index.isin(set(isolated_index))]

        print(f"after filtering, check-in num: {df.shape}")
        return df

    @classmethod
    def generate_id(cls, df, session_time_interval, aoi_dis_interval, do_label_encoder=True, only_last_metric=True):
        # TODO: 自动选取interval参数
        df = df.sort_values(by=['UserId', 'UTCTimeOffset'], ascending=True)

        # generate pseudo session trajectory(temporal)
        start_id = 0
        pseudo_session_trajectory_id = [start_id]
        # query_pseudo_session_trajectory_id = [None]
        start_user = df['UserId'].tolist()[0]
        time_interval = []
        for user, time_diff in tqdm(zip(df['UserId'], df['UTCTimeOffset'].diff())):
            if pd.isna(time_diff):
                time_interval.append(None)
                continue
            elif start_user != user:
                # difference user
                start_id += 1
                start_user = user
            elif time_diff.total_seconds() / 60 > session_time_interval:
                # same user, beyond interval
                start_id += 1
            time_interval.append(time_diff.total_seconds() / 60)
            pseudo_session_trajectory_id.append(start_id)

        # generate pseudo aoi trajectory(spatial)
        start_id = 0
        pseudo_aoi_trajectory_id = [start_id]
        # query_pseudo_aoi_trajectory_id = [None]
        start_user = df.iloc[0]['UserId']
        lat_lng_shift = df[['Latitude', 'Longitude']].shift()
        distance_interval = []
        for user_id, lat, lng, lat_shift, lng_shift in tqdm(zip(
                df['UserId'], df['Latitude'], df['Longitude'], lat_lng_shift['Latitude'], lat_lng_shift['Longitude'])):
            # last_trajectory_id = start_id
            distance = haversine(lng_shift, lat_shift, lng, lat)
            distance_interval.append(distance)
            if pd.isna(distance):
                continue
            elif start_user != user_id:
                start_id += 1
                start_user = user_id
            elif distance >= aoi_dis_interval:
                start_id += 1
            pseudo_aoi_trajectory_id.append(start_id)
            # query_pseudo_aoi_trajectory_id.append(last_trajectory_id)
        assert len(pseudo_session_trajectory_id) == len(df)
        assert len(pseudo_aoi_trajectory_id) == len(df)

        # do label encoding
        if do_label_encoder:
            poi_id_lb = LabelEncoder()
            poi_id_lb = poi_id_lb.fit(df[df['SplitTag'] == 'train']['PoiId'].values.tolist())
            padding_poi_id = 0
            df['PoiId'] = [
                poi_id_lb.transform([i])[0] + 1 if i in poi_id_lb.classes_ else padding_poi_id
                for i in df['PoiId'].values.tolist()
            ]
            poi_category_lb = LabelEncoder()
            poi_category_lb = poi_category_lb.fit(df[df['SplitTag'] == 'train']['PoiCategoryId'].values.tolist())
            padding_poi_category = 0
            df['PoiCategoryId'] = [
                poi_category_lb.transform([i])[0] + 1 if i in poi_category_lb.classes_ else padding_poi_category
                for i in df['PoiCategoryId'].values.tolist()
            ]
            user_id_lb = LabelEncoder()
            user_id_lb = user_id_lb.fit(df[df['SplitTag'] == 'train']['UserId'].values.tolist())
            padding_user_id = 0
            df['UserId'] = [
                user_id_lb.transform([i])[0] + 1 if i in user_id_lb.classes_ else padding_user_id
                for i in df['UserId'].values.tolist()
            ]

            hour_id_lb = LabelEncoder()
            hour_id_lb = hour_id_lb.fit(df[df['SplitTag'] == 'train']['UTCTimeOffsetHour'].values.tolist())
            padding_hour = 0
            df['UTCTimeOffsetHour'] = [
                hour_id_lb.transform([i])[0] + 1 if i in hour_id_lb.classes_ else padding_hour
                for i in df['UTCTimeOffsetHour'].values.tolist()
            ]

            weekday_id_lb = LabelEncoder()
            weekday_id_lb = weekday_id_lb.fit(df[df['SplitTag'] == 'train']['UTCTimeOffsetWeekday'].values.tolist())
            padding_weekday = 0
            df['UTCTimeOffsetWeekday'] = [
                weekday_id_lb.transform([i])[0] + 1 if i in weekday_id_lb.classes_ else padding_weekday
                for i in df['UTCTimeOffsetWeekday'].values.tolist()
            ]

            # df['UserId'] = user_id_lb.transform(df['UserId'].values.tolist())
            with open(osp.join(cls.root_path, 'processed', 'label_encoding.pkl'), 'wb') as f:
                pickle.dump([
                    poi_id_lb, poi_category_lb, user_id_lb, hour_id_lb, weekday_id_lb,
                    padding_poi_id, padding_poi_category, padding_user_id, padding_hour, padding_weekday
                ], f)

        df['check_ins_id'] = df['UTCTimeOffset'].rank(ascending=True, method='first') - 1
        df['time_interval'] = time_interval
        df['pseudo_session_trajectory_id'] = pseudo_session_trajectory_id
        df['distance_interval'] = distance_interval
        df['pseudo_aoi_trajectory_id'] = pseudo_aoi_trajectory_id

        # 按照轨迹id统计rank, 用于ignore轨迹第一个checkin样本, 和GETNext对齐
        df['pseudo_session_trajectory_rank'] = df.groupby(
            'pseudo_session_trajectory_id')['UTCTimeOffset'].rank(method='first')
        df['query_pseudo_session_trajectory_id'] = df['pseudo_session_trajectory_id'].shift()
        df.loc[df['pseudo_session_trajectory_rank'] == 1, 'query_pseudo_session_trajectory_id'] = None
        df['last_checkin_epoch_time'] = df['UTCTimeOffsetEpoch'].shift()
        df.loc[df['pseudo_session_trajectory_rank'] == 1, 'last_checkin_epoch_time'] = None
        df.loc[df['pseudo_session_trajectory_rank'] == 1, 'SplitTag'] = 'ignore'

        # 按照aoi的轨迹id统计rank, 但暂时不做ignore, 会把query trajectory id置为None
        df['pseudo_aoi_trajectory_rank'] = df.groupby(
            'pseudo_aoi_trajectory_id')['UTCTimeOffset'].rank(method='first')
        df['query_pseudo_aoi_trajectory_id'] = df['pseudo_aoi_trajectory_id'].shift()
        df.loc[df['pseudo_aoi_trajectory_rank'] == 1, 'query_pseudo_aoi_trajectory_id'] = None

        if only_last_metric:
            df['pseudo_session_trajectory_count'] = df.groupby(
                'pseudo_session_trajectory_id')['UTCTimeOffset'].transform('count')
            df.loc[(df['SplitTag'] == 'validation') & (
                    df['pseudo_session_trajectory_count'] != df['pseudo_session_trajectory_rank']
            ), 'SplitTag'] = 'ignore'
            df.loc[(df['SplitTag'] == 'test') & (
                    df['pseudo_session_trajectory_count'] != df['pseudo_session_trajectory_rank']
            ), 'SplitTag'] = 'ignore'

        print(f'ignore sample num: {len(df[df["SplitTag"] == "ignore"])}')

        trajectory_id_count = df.groupby(['pseudo_session_trajectory_id'])['check_ins_id'].count().reset_index()
        check_ins_count = trajectory_id_count[trajectory_id_count['check_ins_id'] == 1]
        # print(df.loc[(df['pseudo_session_trajectory_id'].isin(check_ins_count['pseudo_session_trajectory_id'].values.tolist())) & (df['SplitTag'] != 'ignore')].shape)

        # tmp = trajectory_id_count[trajectory_id_count['check_ins_id'] > 1]
        # df = df.merge(tmp[['pseudo_session_trajectory_id']], on=["pseudo_session_trajectory_id"])

        print(check_ins_count)
        print(f"pseudo session trajectory of single check-ins count: {len(check_ins_count)}, "
              f"ratio: {len(check_ins_count) / len(trajectory_id_count)}")
        trajectory_id_count = df.groupby(['pseudo_aoi_trajectory_id'])['check_ins_id'].count().reset_index()
        check_ins_count = trajectory_id_count[trajectory_id_count['check_ins_id'] == 1]
        print(f"pseudo aoi trajectory of single check-ins count: {len(check_ins_count)}, "
              f"ratio: {len(check_ins_count) / len(trajectory_id_count)}")

        cols = ['check_ins_id', 'UTCTimeOffset', 'UTCTimeOffsetEpoch', 'pseudo_session_trajectory_id',
                'query_pseudo_session_trajectory_id', 'pseudo_aoi_trajectory_id', 'query_pseudo_aoi_trajectory_id',
                'UserId', 'Latitude', 'Longitude', 'PoiId', 'PoiCategoryId', 'PoiCategoryName',
                'last_checkin_epoch_time']
        print(f'before save data, shape: {df.shape}')
        print(
            f"validation shape: {df[df['SplitTag'] == 'validation'].shape}",
            f"test shape: {df[df['SplitTag'] == 'test'].shape}")
        df[df['SplitTag'] == 'train'][cols].to_csv(
            osp.join(cls.root_path, 'processed', 'train_sample.csv'), index=False)
        df[df['SplitTag'] == 'validation'][cols].to_csv(
            osp.join(cls.root_path, 'processed', 'validation_sample.csv'), index=False)
        df[df['SplitTag'] == 'test'][cols].to_csv(
            osp.join(cls.root_path, 'processed', 'test_sample.csv'), index=False)
        return df

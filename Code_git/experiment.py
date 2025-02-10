from Code.model import * 
from sklearn.metrics import root_mean_squared_error
from Code.utility import GridSearcher
import sys 
from Code.dataloader import * 
import pickle
import time 

class Experiment:
    def __init__(self, result_save_p, ml_params = None):
        self.result_save_p = result_save_p
        self.ml_params = ml_params
        # result log
        self.datalist = ['anemones', 'bronzefilter', 'longleaf', 'spruces', 'waka']
        self.p = 'Your Own Path'
        self.xy_data_paths = [self.p + i + '.csv' for i in self.datalist]
        self.dataload_ls = [ load_london, load_melbourne, load_newyork, load_paris, load_beijing, load_perth, load_seattle, load_dubai ]
        self.data_n = [  'london', 'melbourne', 'newyork', 'paris', 'beijing', 'perth', 'seattle', 'dubai' ]
       
        self.rst = {i :{} for i in self.data_n}
        self.rst.update({i + '_coords': {} for i in self.data_n})
        self.rst.update({i + '_coords': {} for i in self.datalist})

    def load_data(self, loader = None, path = None, coords_only = False):
        if loader == load_xy_only_data:
            df = loader(path, split_rate=(0.7, 0.1, 0.2), scale =True)
        else:
            df = loader(split_rate=(0.7, 0.1, 0.2), scale =True, coords_only = coords_only)
        return df 

    def  ml_run(self, df_label, df):
        score_f = root_mean_squared_error
        # prepare data 
        df_train_raw = df[df['split_type'] == 0].sample(frac=1).reset_index(drop=True)
        y_train = df_train_raw['regression_target']
        df_train_raw = df_train_raw.drop(['regression_target', 'split_type'], axis=1)

        df_val_raw = df[df['split_type'] == 1].sample(frac=1).reset_index(drop=True)
        y_val = df_val_raw['regression_target']
        df_val_raw = df_val_raw.drop(['regression_target', 'split_type'], axis=1)

        df_test_raw = df[df['split_type'] == 2]
        y_test = df_test_raw['regression_target']
        df_test_raw = df_test_raw.drop(['regression_target', 'split_type'], axis=1)
        train_size = len(df_train_raw)
        data_size = len(df)
        
        for k, params in self.ml_params.items():
            print(k)
            regressor = params['regressor']
            searching_param = params['searching_param']
            search_num = 1
            # dataset 
            if k in ['GWR', 'Kriging', 'Kriging_LGBMRegressor']:
                df_train = df_train_raw.drop(['x', 'y'], axis=1)
                df_val = df_val_raw.drop(['x', 'y'], axis=1)
                df_test = df_test_raw.drop(['x', 'y'], axis=1)
            else:
                df_train = df_train_raw.drop(['lat', 'lon'], axis=1)
                df_val = df_val_raw.drop(['lat', 'lon'], axis=1)
                df_test = df_test_raw.drop(['lat', 'lon'], axis=1)

            s_t = time.time()
            if searching_param is not None:
                try:
                    searcher = GridSearcher(searching_param, score_f)
                except:
                    print("error:", sys.exc_info()[0])
                    pass 
                if k not in ['LightGBM', 'XGBoost', 'CatBoost']:
                    best_score, best_param, best_model, search_num = searcher.search(regressor, df_train, df_val, y_train, y_val)
                else:
                    best_score, best_param, best_model, search_num = searcher.search_tree_ensemble(regressor, df_train, df_val, y_train, y_val, choice = k)
            else:
                regressor.fit(df_train, y_train)
                best_model = regressor
                best_param = None 
            e_t = time.time()
            pred = best_model.predict(df_test)
            rst = score_f(y_test, pred)
            self.rst[df_label][k] = (pred, y_test, rst, best_param, e_t - s_t, search_num, train_size, data_size)
            with open(self.result_save_p + df_label + '_performance.pkl', 'wb') as file:
                pickle.dump(self.rst, file)
        return 'ml done!'

    def main(self):
        print('experiment begins!')
        # run estate 
        for ix in range(len(self.data_n)):
            print(self.data_n[ix], ' begins!')
            loader = self.dataload_ls[ix]
            df = self.load_data(loader, coords_only = False)
            self.ml_run(self.data_n[ix], df)
        for ix in range(len(self.data_n)):
            loader = self.dataload_ls[ix]
            df = self.load_data(loader, coords_only = True)
            print(self.data_n[ix] + '_coords', ' begins!')
            self.ml_run(self.data_n[ix] + '_coords', df)
        
        # run coords only 
        for ix in range(len(self.datalist)):
            loader = load_xy_only_data
            df = self.load_data(loader, path = self.xy_data_paths[ix])
            print(self.datalist[ix] + '_coords', ' begins!')
            self.ml_run(self.datalist[ix] + '_coords', df)
        

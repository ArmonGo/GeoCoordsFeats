
import numpy as np 
import osmnx as ox 
import pyproj
import copy
from sklearn.model_selection import ParameterGrid
import lightgbm as lgb 
import xgboost as xgb 
import math 

def get_local_crs(lat, lon, radius):  
    trans = ox.utils_geo.bbox_from_point((lat, lon), dist = radius, project_utm = True, return_crs = True)
    to_csr = pyproj.CRS( trans[-1])
    return to_csr

def haversine(coord1, coord2):
    R = 6371.0  # Radius of Earth in kilometers
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# Function to find the centroid and radius
def find_centroid_and_radius(coords):
    # Calculate the centroid (average latitude and longitude)
    avg_lat = sum(coord[0] for coord in coords) / len(coords)
    avg_lon = sum(coord[1] for coord in coords) / len(coords)
    centroid = (avg_lat, avg_lon)
    # Calculate the radius (maximum distance from centroid to any point)
    radius = max(haversine(centroid, coord) for coord in coords) * 1000
    return centroid, np.ceil(radius) + 5000 # add buffer 


class GridSearcher:
    def __init__(self, grid, score_f):
        self.param_grid =  ParameterGrid(grid)
        self.score_f = score_f
        self.best_score = np.inf
        self.best_param = None
        self.best_model = None 

    def search(self, rg, df_train, df_val, y_train, y_val):
        for param in self.param_grid:
            input_p = copy.deepcopy(param)
            regressor = rg() # instantiate the regressor 
            regressor.set_params(**input_p)
            regressor.fit(df_train, y_train)
            # count score
            pred = regressor.predict(df_val)
            s = self.score_f(y_val, pred)
            if s < self.best_score:
                self.best_score = s
                self.best_param = param
                self.best_model = copy.deepcopy(regressor)
        return self.best_score, self.best_param, self.best_model, len(list(self.param_grid))
    
    def search_tree_ensemble(self, rg, df_train, df_val, y_train, y_val, choice):
        for param in self.param_grid:
            regressor = rg() # instantiate the regressor 
            regressor.set_params(**param)
            if choice == 'LightGBM':
                callbacks = lgb.early_stopping(stopping_rounds = 100, first_metric_only = True, verbose = False, min_delta=0.0)
                regressor.fit(df_train, y_train, callbacks = [callbacks],  # use the best iteration to predict
                                eval_metric ='rmse', eval_set =[(df_val, y_val)])
            elif choice == 'XGBoost':
                callbacks = xgb.callback.EarlyStopping(rounds=100, metric_name='rmse', data_name='validation_0', save_best=True)  
                regressor.set_params(**{'callbacks' : [callbacks]})
                regressor.fit(df_train, y_train,  # return the best model
                             eval_set =[(df_val, y_val)],
                             verbose=False)
            elif choice == 'CatBoost':
                regressor.set_params(**param)
                regressor.fit(df_train, y_train,  # return the best model
                             eval_set =[(df_val, y_val)],
                             use_best_model=True,
                            early_stopping_rounds=100,
                            verbose_eval = False)
            # count score
            pred = regressor.predict(df_val)
            s = self.score_f(y_val, pred)
            if s < self.best_score:
                self.best_score = s
                self.best_param = param
                self.best_model = copy.deepcopy(regressor)
        return self.best_score, self.best_param, self.best_model, len(list(self.param_grid))





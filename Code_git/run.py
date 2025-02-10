from experiment import Experiment
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from Code.model import GWR, KrigeRegressor, OrdinaryKriging
import numpy as np 
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.linear_model import TweedieRegressor
from catboost import CatBoostRegressor
from tabpfn import TabPFNRegressor


result_save_p = './Results/'

ml_params = {   'GWR' : {'regressor' : GWR, 
                        'searching_param' : {"constant": [True]}  
                         },
                'Kriging_LGBMRegressor' : {'regressor' : KrigeRegressor, 
                             'searching_param' :   {"outer_params" : list(ParameterGrid([{ "nlags" : range(30,150,30),  #range(20,120,20),
                                                                                            "variogram_model": [ "linear", "gaussian"] ,
                                                                                            'regression_model': [LGBMRegressor]}])),  # outer model params
                                                    'inner_params' : list(ParameterGrid([ { "reg_alpha" : np.arange(0, 1.5, 0.5),
                                                                                            "reg_lambda" : np.arange(0, 1.5, 0.5),
                                                                                            "learning_rate" : [0.1, 0.01, 0.005],
                                                                                            "verbose": [-100]
                                                                                            } ]))
                                                    } 
                       },
                "Kriging": { 'regressor' : OrdinaryKriging, 
                            "searching_param": {"nlags" : range(30,150,30),
                                                "variogram_model":[ "gaussian", "linear"] } 
                                },
                'Lr_ridge' : {'regressor' : Ridge, 
                        'searching_param' : {"alpha": np.arange(0.1,1,0.1)}
                        },
                'RandomForest' : {'regressor' : RandomForestRegressor, 
                        'searching_param' :  {
                                              "min_samples_split" : [2,3,5],
                                              "min_samples_leaf" : [3,5,10]
                                             }
                                 },     
                'XGBoost' : {'regressor' : XGBRegressor, 
                        'searching_param' :  { "learning_rate" : [0.1, 0.01, 0.005],
                                                "reg_alpha" : np.arange(0, 1.1, 0.1),
                                                "reg_lambda" : np.arange(0, 1.1, 0.1)
                                            } 
                                 },
                'LightGBM' : {'regressor' : LGBMRegressor, 
                                'searching_param' : {
                                                    "reg_alpha" : np.arange(0, 1.1, 0.1),
                                                    "reg_lambda" : np.arange(0, 1.1, 0.1),
                                                    "learning_rate" : [0.1, 0.01, 0.005],
                                                    "verbose": [-100]
                                                    }     
                             },
                
                "SVM":  { 'regressor' : SVR, 
                            "searching_param": {"C": range(1,105,10),
                               "epsilon": np.arange(0.1, 1, 0.1)}
                        },
                "Guassian":  { 'regressor' : GaussianProcessRegressor, 
                            "searching_param": {"kernel":  [ C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))],
                               "alpha" : np.arange(0.1, 1, 0.1)
                                                }
                                },         
                "TweedieRegressor":  { 'regressor' : TweedieRegressor, 
                            "searching_param":  {'power' : [0, 1,1.2, 1.5, 1.8, 2, 3],
                                                  'alpha' :  list(np.arange(0, 1.0, 0.1)) + [2, 5, 8, 10]}
                                },
                "CatBoost" :  {'regressor' : CatBoostRegressor, 
                            "searching_param":  { 'iterations': [100, 200],
                                                'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
                                                'l2_leaf_reg': [0.1, 0.5, 1, 5]}},
                "TabPFNRegressor" : {'regressor': TabPFNRegressor(ignore_pretraining_limits = True),
                                       "searching_param":  None
                                      
                }
}


if __name__ == "__main__":
    exp = Experiment(result_save_p, ml_params)
    exp.main()


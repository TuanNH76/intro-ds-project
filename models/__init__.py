from .random_forest import MyRandomForest
from .sarimax import MySARIMAX
from .LSTM import MyLSTM
from .GRU import MyGRU
from .arima import MyARIMA
from .xgboost import MyXGBoost

MODELS = {
    'random_forest': MyRandomForest,
    'sarimax': MySARIMAX,
    'lstm': MyLSTM,
    'gru': MyGRU,
    'arima': MyARIMA,
    'xgboost': MyXGBoost,
}
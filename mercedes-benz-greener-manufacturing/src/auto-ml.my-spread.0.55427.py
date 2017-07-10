import pandas as pd
from auto_ml import Predictor
from auto_ml.utils_models import load_ml_model

train = pd.read_csv('../input/my_spread_train.csv')
test = pd.read_csv('../input/my_spread_test.csv')

column_descriptions = { 'y': 'output' }
ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)
# ml_predictor.train(train, model_names=['DeepLearningRegressor', 'XGBRegressor'])
ml_predictor.train(train)
print ml_predictor.score(train, train.y)

test['y'] = ml_predictor.predict(test)
test.loc[:,['ID', 'y']].to_csv('../output/auto-ml.my-spread.csv', index=False)

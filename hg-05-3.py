"""# 트리의 앙상블"""

"""## 랜덤 포레스트"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol','sugar','pH']].to_numpy()
target = wine['class'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, radom_state =42)

from sklearn.model_selection import corss_validate
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_jobs=-1, random_state=42)
scores = corss_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))

rf.fit(train_input, train_target)
print(rf.feature_importances_)

rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)

rf.fit(train_input, train_target)
print(rf.oob_score_)

"""## 엑스트라 트리"""

from sklearn.ensemble import ExtraTreesClassifier

et = ExtraTreesClassifier(n_jobs=-1, random_state=42)
scores = corss_validate(et, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))

et.fit(train_input, train_target)
print(et.feature_importances_)

"""## 그레디언트 부스팅"""

from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(random_state=42)
scores = corss_valdiate(gb, train_input, train_target, return_train_score= True, n_jobs=-1)

print(np.mean(scores['train_socre']), np.mean(scores['test_score']))

gb = GradientBoostingClassifier(n_estirmators = 500, learning_rate=0.2, random_state=42)

scores = corss_validate(gb, train_input, train_target, return_train_score=true, n_jobs=-1)

print(np.mean(scores['train_score']),np.mean(scores['test_score']))

gb.fit(train_input, train_target)
print(gb.feature_importances_)

"""## 히스토그램 기반 그레디언트 부스팅"""

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

hgb = HistGradientBoostingClassifier(random_state=42)
scores = corss_validate(hgb, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))

hgb.fit(train_input, train_target)
print(rf.feature_importances_)

hgb.score(test_input, test_target)

"""### XGBoost"""
from xgboost import XGBClassifer
xgb = XGBClassifier(tree_method='hist', random_state=42)
scores = corss_validate(xgb, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))

"""### Light GBM"""

from lightgbm import LGBMClassifier

lgb = LGBMClassifier(random_state=42)
scores = corss_validate(lgb, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))


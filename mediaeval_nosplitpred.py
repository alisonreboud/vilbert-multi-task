
import time
import string 
import pickle
import itertools
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# from nltk.corpus import stopwords
from sklearn.linear_model import SGDClassifier, SGDRegressor, LogisticRegression, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from scipy.stats import spearmanr
#from tqdm.notebook import tqdm
#from nltk.tokenize import RegexpTokenizer
#from gensim.models import KeyedVectors


with open('datasets/ME2020/out_features/train_features_avMEVQAframe.pkl', 'rb') as f:
    data = (pickle.load(f))['pooled_output_mul']
    #print(len(data['pooled_output_mul'][0]))
with open('datasets/ME2020/out_features/dev_features6framesMEVQA.pkl', 'rb') as f:
    data_dev = (pickle.load(f))['pooled_output_mul']
    #print(len(data_dev['pooled_output_mul'][0]))
with open('Surrey/big_segments.pkl', 'rb') as f:
    data_test = (pickle.load(f))['pooled_output_mul']



df_data = pd.read_csv('scores_v2.csv' )
scores_dev=pd.read_csv('dev_scores.csv')
X_vilbert = []
#Y = []
#for i, entry in df_data.iterrows():
    #y = (entry['part_1_scores'], entry['part_2_scores'])
    #Y.append(y)

def enumerate_models(models):
    instances = []
    for model_name, (model, hyperparameters) in models.items():
        configs = {}
        if len(hyperparameters) > 0:
            params, vals = list(hyperparameters.keys()), list(hyperparameters.values())
            configs = [dict(zip(params, vv)) for vv in list(itertools.product(*vals))]
            for config in configs:
                m = model(**config)
                instances.append(m)
        else:
            instances.append(model())
    return instances

regression_models = {
    #'LogisticRegression': (LogisticRegression, {"C": [1e3, 1, 1e-3], "penalty": ['l1', 'l2', 'elasticnet']}),
    #'LinearRegression': (LinearRegression, {}),
    #'MLPRegressor': (MLPRegressor, {'alpha': [1e-3, 1e-5, 1e-7], 'hidden_layer_sizes': [(10,), (50,), (100,)]}),
    #'SGDRegressor': (SGDRegressor, {'alpha': [0.00001, 0.0001, 0.1,]}),
    #'SVR': (SVR, {'kernel': ['linear', 'poly', 'rbf'], "C": [1e3, 1., 1e-3]})
    'SVR': (SVR, {'kernel': ['rbf'], "C": [ 1e-3]})
}

#X =  {'pooled_output_mul': data['pooled_output_mul']}
X_train=data
X_test=data_test
Y_train_st = df_data['part_1_scores']
Y_train_lt = df_data['part_2_scores']
Y_id=df_data['video_id']
Y_test_st = scores_dev['part_1_scores']
Y_test_lt = scores_dev['part_2_scores']
spearman = lambda x,y: spearmanr(x, y).correlation
"""
results_st = {}
#for k in range(len(X)):
 #print(X['targets'][k])
for k in X_train:
 results_st[k] = {}
 for regressor in enumerate_models(regression_models):
  model_name = str(regressor)
  results_st[k][model_name] = []
  print('Training', model_name, '..')
  t = time.time()
  X_train_k, X_test_k = X_train[k], X_test[k]
  #id_train, id_test = Y_id[train_index], Y_id[test_index]
  regressor.fit(X_train_k.cpu(), Y_train_st)
  y_pred = regressor.predict(X_test_k.cpu())
  #print(y_pred)
  print(y_pred)
  #print(spearman(y_pred, Y_test_st))
  #results_st[k][model_name].append((y_pred, Y_test_st))

  print(f'done! ({time.time() - t:2} secs)')
  t = time.time()
"""


#for k in range(len(X)):
 #print(X['targets'][k])
results_st= []
#id_train, id_test = Y_id[train_index], Y_id[test_index]
for regressor in enumerate_models(regression_models):
 regressor.fit(X_train.cpu(), Y_train_st)
 y_pred = regressor.predict(X_test.cpu())
 print(y_pred)
 s = pd.Series(y_pred)
 print(s)
 s.to_csv('Surrey/st_surrey_vilbert_big_segments.csv')
 #print(spearman(y_pred, Y_test_st))
  #results_st[k][model_name].append((y_pred, Y_test_st))

  #print(f'done! ({time.time() - t:2} secs)')
  #t = time.time()


#spearman = lambda x,y: spearmanr(x, y).correlation

#results_st=pd.DataFrame(columns=['prediction','gt','id'])
#resuls_lt=pd.DataFrame(columns=['prediction','gt'])

#for term, all_folds in [('Short term', folds), ('Long term', folds_lt)]:
#for term, embedding in [('Short term', results_st)]:
    #print(term.upper())
    #print(embedding)
    #print('USING', embedding, ':')
    #for model_name in embedding:
    #print(embedding)
     #print(model_name.split('(')[0], '     ', model_name.split('(')[1][:-1][:50])
     # print(', '.join([str(spearman(y_p, y_t)) for y_p, y_t in all_folds[embedding][model_name]]))
     #print(round(sum([spearman(y_p, y_t) for y_p, y_t in embedding[model_name][0])/len(embedding[model_name]), 3))
            #if embedding == 'pooled_output_mul' and term == 'Short term':
             #for y_p, y_t,id_test in all_folds[embedding][model_name]:
              #data=pd.DataFrame({'prediction': y_p,'gt':y_t,'id':id_test})
             # print(all_folds[embedding][model_name])
              #with open('6folds_st.pkl', 'wb') as f:
               #pickle.dump(all_folds[embedding][model_name],f)
            #if embedding == 'pooled_output_v' and term == 'Long term':
             #for y_p, y_t,id_test in all_folds[embedding][model_name]:
              #data=pd.DataFrame({'prediction': y_p,'gt':y_t,'id':id_test})
              #print(all_folds[embedding][model_name])
              #with open('6folds_lt.pkl', 'wb') as f:
               #pickle.dump(all_folds[embedding][model_name],f)

             #results_st=pd.DataFrame(columns=['pred','gt','id'],data=all_folds[embedding][model_name])
             #print(y_p for y_p, y_t in all_folds[embedding][model_name]])/len(all_folds[embedding][model_name])
#print(results_st)



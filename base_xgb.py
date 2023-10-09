import math
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from datetime import datetime
from gensim.models import Word2Vec
import xgboost as xgb
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

train = pd.read_csv('训练集/train.txt', header=None, names=['pid', 'label', 'gender', 'age', 'tagid', 'time', 'province', 'city', 'make', 'model'])
test = pd.read_csv('测试集/apply_new.txt', header=None, names=['pid', 'gender', 'age', 'tagid', 'time', 'province', 'city', 'make', 'model'])
data = pd.concat([train, test])

data['tagid'] = data['tagid'].apply(lambda x: eval(x))
sentences = data['tagid'].values.tolist()
for i in range(len(sentences)):
    sentences[i] = [str(x) for x in sentences[i]]

emb_size = 32
model = Word2Vec(sentences, vector_size=emb_size, window=6, min_count=5, sg=0, hs=0, seed=1, epochs=5)
model.save('word2vec_model.word2vec')  # save word2vec model

emb_matrix = []
for seq in sentences:
    vec = []
    for w in seq:
        if w in model.wv.key_to_index:
            vec.append(model.wv[w])
    if len(vec) > 0:
        emb_matrix.append(np.mean(vec, axis=0))
    else:
        emb_matrix.append([0] * emb_size)
emb_matrix = np.array(emb_matrix)
for i in range(emb_size):
    data['tag_emb_{}'.format(i)] = emb_matrix[:, i]

cat_cols = ['gender', 'age', 'province', 'city']
features = [i for i in data.columns if i not in ['pid', 'label', 'tagid', 'time', 'model', 'make']]

data[cat_cols] = data[cat_cols].astype('category')
X_train = data[~data['label'].isna()]
X_test = data[data['label'].isna()]

y = X_train['label']
KF = StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
params = {
          'objective':'binary',
          'metric':'binary_error',
          'learning_rate':0.05,
          'subsample':0.8,
          'subsample_freq':3,
          'colsample_btree':0.8,
          'num_iterations': 10000,
          'verbose':-1
}
oof_lgb = np.zeros(len(X_train))
predictions_lgb = np.zeros((len(X_test)))
# 特征重要性
feat_imp_df = pd.DataFrame({'feat': features, 'imp': 0})
# 五折交叉验证
for fold_, (trn_idx, val_idx) in enumerate(KF.split(X_train.values, y.values)):
    print("fold n°{}".format(fold_))
    print('trn_idx:',trn_idx)
    print('val_idx:',val_idx)
    # trn_data = lgb.Dataset(X_train.iloc[trn_idx][features],label=y.iloc[trn_idx])
    # val_data = lgb.Dataset(X_train.iloc[val_idx][features],label=y.iloc[val_idx])
    # trn_data = xgb.DMatrix(X_train.iloc[trn_idx][features], label=y.iloc[trn_idx])
    # val_data = xgb.DMatrix(X_train.iloc[val_idx][features], label=y.iloc[val_idx])
    num_round = 10000
    # clf = lgb.train(
    #     params,
    #     trn_data,
    #     num_round,
    #     valid_sets = [trn_data, val_data],
    #     verbose_eval=100,
    #     early_stopping_rounds=50,
    #     categorical_feature=cat_cols,
    # )
    aa = X_train.iloc[trn_idx][features]
    bb = y.iloc[trn_idx]
    aaa = X_train.iloc[trn_idx][features]._get_numeric_data()
    bbb = y.iloc[trn_idx]._get_numeric_data()

    clf = XGBClassifier(
        max_depth=4,
        n_estimators=50,
        eval_metric='error',
        scale_pos_weight=1,
        use_label_encoder=False,
        min_child_weight=5,
        subsample=0.8)
    clf.fit(X_train.iloc[trn_idx][features], y.iloc[trn_idx])

    # feat_imp_df['imp'] += clf.feature_importance() / 5
    # oof_lgb[val_idx] = clf.predict(X_train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    # predictions_lgb[:] += clf.predict(X_test[features], num_iteration=clf.best_iteration)
    oof_lgb[val_idx] = clf.predict(X_train.iloc[val_idx][features])
    predictions_lgb[:] += clf.predict(X_test[features])

print("AUC score: {}".format(roc_auc_score(y, oof_lgb)))
print("F1 score: {}".format(f1_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
print("Precision score: {}".format(precision_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
print("Recall score: {}".format(recall_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))

X_test['category_id'] = [1 if i >= 2.5 else 0 for i in predictions_lgb]
X_test['user_id'] = X_test['pid']
X_test[['user_id', 'category_id']].to_csv('base_sub.csv', index=False)
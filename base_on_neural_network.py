# coding:utf-8
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
import tensorflow as tf
import warnings
from gensim.models import Word2Vec


##########################################  加载数据   ######################################################
warnings.filterwarnings('ignore')

# 读取数据，简单处理list数据
train = pd.read_csv('./训练集/train.txt', header=None)
test = pd.read_csv('./测试集/apply_new.txt', header=None)

train.columns = ['pid', 'label', 'gender', 'age', 'tagid', 'time', 'province', 'city', 'model', 'make']
test.columns = ['pid', 'gender', 'age', 'tagid', 'time', 'province', 'city', 'model', 'make']

train['label'] = train['label'].astype(int)

data = pd.concat([train, test])
data['label'] = data['label'].fillna(-1)

data['tagid'] = data['tagid'].apply(lambda x: eval(x))
data['tagid'] = data['tagid'].apply(lambda x: [str(i) for i in x])


#####################################   构建词库    ############################################
# 超参数
# embed_size  embedding size
# MAX_NB_WORDS  tagid中的单词出现次数
# MAX_SEQUENCE_LENGTH  输入tagid list的长度
embed_size = 64
MAX_NB_WORDS = 230637
MAX_SEQUENCE_LENGTH = 128
# 训练word2vec，这里可以考虑elmo，bert等预训练
w2v_model = Word2Vec(sentences=data['tagid'].tolist(), vector_size=embed_size, window=5, min_count=1, epochs=10)
# 这里是划分训练集和测试数据
X_train = data[:train.shape[0]]['tagid']
X_test = data[train.shape[0]:]['tagid']

# 创建词典，利用了tf.keras的API，编码
tokenizer = text.Tokenizer(num_words=MAX_NB_WORDS)  # 分词器
tokenizer.fit_on_texts(list(X_train) + list(X_test))  # fit_on_text(texts) 使用一系列文档来生成token词典，texts为list类，每个元素为一个文档
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
X_train = sequence.pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
X_test = sequence.pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
word_index = tokenizer.word_index
# 计算一共出现了多少个单词
# 把每一个tagid看成是一个单词

nb_words = len(word_index) + 1
print('Total %s word vectors.' % nb_words)
# 构建一个embedding的矩阵，之后输入到模型使用
# 230638行64列，每一个tagid对应一个64维向量
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    try:
        embedding_vector = w2v_model.wv.get_vector(word)
    except KeyError:
        continue
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

y_categorical = train['label'].values



##########################################   定义模型        ####################################################
def my_model():
    embedding_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    # 词嵌入（使用预训练的词向量）

    # keras中的Embedding和Word2vec的区别
    # 其实二者的目标是一样的，都是为了学到词的稠密的嵌入表示。只不过学习的方式不一样。
    # Word2vec是无监督的学习方式，利用上下文环境来学习词的嵌入表示，因此可以学到相关词，但是只能捕捉到局部分布信息。
    # 而在keras的Embedding层中，权重的更新是基于标签的信息进行学习，为了达到较高的监督学习的效果，会将Embedding作为网络的一层，
    # 根据target进行学习和调整。比如LSTM中对词向量的微调。简单来说，Word2vec一般单独提前训练好，
    # 而Embedding一般作为模型中的层随着模型一同训练。
    embedder = Embedding(nb_words,
                         embed_size,
                         input_length=MAX_SEQUENCE_LENGTH,
                         weights=[embedding_matrix],
                         trainable=False
                         )
    embed = embedder(embedding_input)
    l = LSTM(128)(embed)  # 128是size
    flat = BatchNormalization()(l)
    drop = Dropout(0.2)(flat)
    main_output = Dense(1, activation='sigmoid')(drop)
    model = Model(inputs=embedding_input, outputs=main_output)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=['accuracy'])
    return model


###########################################   模型训练     ###################################################
# 五折交叉验证
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=666)
oof = np.zeros([len(train), 1])
predictions = np.zeros([len(test), 1])

######################
predictions_save = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
    print("fold n{}".format(fold_ + 1))
    model = my_model()
    if fold_ == 0:
        model.summary()

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3)
#     bst_model_path = "./lstm_{epoch:04d}.ckpt"
    bst_model_path = "./trainning/lstm_{}.model".format(fold_)
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

    X_tra, X_val = X_train[trn_idx], X_train[val_idx]
    y_tra, y_val = y_categorical[trn_idx], y_categorical[val_idx]

    model.fit(X_tra, y_tra,
              validation_data=(X_val, y_val),
              epochs=128, batch_size=256, shuffle=True,
              callbacks=[early_stopping, model_checkpoint])

    model.load_weights(bst_model_path)

    oof[val_idx] = model.predict(X_val)

    predictions += model.predict(X_test) / folds.n_splits
    print(predictions)
    df_predictions = pd.DataFrame(predictions)
    predictions_save = predictions_save.append(df_predictions)
    del model

predictions_save.to_csv(r'predictions_save.csv',index=False, header=False)

train['predict'] = oof
train['rank'] = train['predict'].rank()
train['p'] = 1
train.loc[train['rank'] <= train.shape[0] * 0.5, 'p'] = 0
bst_f1_tmp = round(f1_score(train['label'].values, train['p'].values),4)
print(bst_f1_tmp)

submit = test[['pid']]
submit['tmp'] = predictions
submit.columns = ['user_id', 'tmp']

submit['rank'] = submit['tmp'].rank()
submit['category_id'] = 1
submit.loc[submit['rank'] <= int(submit.shape[0] * 0.5), 'category_id'] = 0

print(submit['category_id'].mean())

submit[['user_id', 'category_id']].to_csv('./submit/submit_{}.csv'.format(str(bst_f1_tmp).split('.')[1]), index=False)


model = my_model()

model.load_weights(bst_model_path)
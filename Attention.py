# encoding: utf-8
"""
@author: xiong
@contact: 
@software: PyCharm
@file: LSTM-ATT.py
@time: 2020/11/17 10:29
"""
#导入必要的库
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from keras.layers import *
from keras.models import *
import keras.backend as K
# np.random.seed(100)
 #设置神经网络参数
lstm_units=64#lstm神经元个数
dataset = pd.read_csv('wind3min.csv', header=0, index_col=0)
dataset = dataset.values
# 将整型变为float
dataset = dataset.astype('float32')
# 归一化处理
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
def creat_train(data,lookback,step):
    timeX,featureX,dataY = [],[],[]
    for i in range(len(data) - lookback-step ):
        timeX.append(data[i:(i + lookback),[0]])
        featureX.append(data[i:i + lookback+step-1, [0,1,2,3,4,6]])
        dataY.append(data[i + lookback+step-1,[0]])
    return np.array(timeX),np.array(featureX), np.array(dataY)
#60步一小时
time_train, x_train, y_train=creat_train(dataset[:10080],20,1)
time_test, x_test, y_test=creat_train(dataset[10080:13440],20,1)
print('训练集测试集shape：', x_train.shape, time_train.shape, y_train.shape, x_test.shape, time_test.shape, y_test.shape)

# 建立模型
SINGLE_ATTENTION_VECTOR = False
def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    TIME_STEPS = int(inputs.shape[1])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul
import time
time_start = time.time()  # 开始计时

inputs=Input(shape=(x_train.shape[1],x_train.shape[2]))
attention_mul = attention_3d_block(inputs)
lstm1=LSTM(lstm_units,activation='relu', return_sequences=False)(attention_mul)

# lstm2=LSTM(lstm_units,activation='relu', return_sequences=False)(lstm1)
# # 在LSTM之后使用Attention
# inputs=Input(shape=(x_train.shape[1],x_train.shape[2]))
# lstm_inputs=Permute([2,1])(inputs)
# lstm1=LSTM(lstm_units,activation='relu', return_sequences=True)(lstm_inputs)
# Att=Permute([2,1])(lstm1)#置换维度
# Attention=Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(Att)
# attention=Dense(x_train.shape[2], activation='sigmoid', name='attention_vec')(Attention)#求解Attention权重
# attention=Activation('softmax',name='attention_weight')(attention)
# weight=Multiply()([Attention, attention])#attention与LSTM对应数值相乘

outputs = Dense(1, activation='relu')(lstm1)
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='mae',optimizer='adam')
model.summary()#展示模型结构
'''
如果直接加载模型，则不需要运行这一段训练
'''
#训练模型
history=model.fit(x_train, y_train, steps_per_epoch = 100, epochs = 6,shuffle = False,
                  validation_data = ( x_test, y_test),
                  validation_steps = (13440 - 10080 - 20)) #训练模型epoch次


time_end = time.time()  # 结束计时
time_c = time_end - time_start  # 运行所花时间
print('time cost', time_c, 's')
model.save('ji_A-LSTM.h5')#保存模型
# # 加载模型
# from keras.models import load_model
# model=load_model('ji_A-LSTM.h5')
# # 获得网络权重
# weights = np.array(model.get_weights())
# print(weights)
# # #输出attention层权重
# attention_layer_model = Model(inputs=model.input,outputs=model.get_layer('attention_vec').output)
# attention_weight = attention_layer_model.predict(x_train)
# attention_weight_final=np.mean(np.array(attention_weight), axis=0)
# pd.DataFrame(attention_weight_final, columns=['attention (%)']).plot(kind='bar',
#                                                                          title='Attention Mechanism as '
#                                                                                'a function of input'
#                                                                                ' dimensions.')
# plt.show()
# #迭代图像
# plt.plot(history.history['loss'],label='train')
# plt.plot(history.history['val_loss'],label='test')
# plt.legend()
# plt.show()
#在训练集上的拟合结果
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
y_train_predict=model.predict(x_train)
plt.plot(y_train)
plt.plot(y_train_predict[1:])
plt.legend(['trainY','trainPredict'])
plt.show()
#输出结果
print('训练集上的MAE/RMSE/R2')
print(mean_absolute_error(y_train_predict, y_train))
print(np.sqrt(mean_squared_error(y_train_predict, y_train)))
print(r2_score(y_train_predict, y_train))
#在测试集上的预测
time_start1 = time.time()
y_test_predict=model.predict(x_test)
time_end1 = time.time()  # 结束计时
time_1 = time_end1 - time_start1  # 运行所花时间
print('time cost', time_1, 's')
plt.plot(y_test)
plt.plot(y_test_predict[1:])
plt.legend(['trainY','trainPredict'])
plt.show()
# 保存预测值
dt = pd.DataFrame(y_test_predict)
dt.to_csv("pre1.csv", index=0)
df = pd.DataFrame(y_test)
df.to_csv("pre2.csv", index=0)

#输出结果
print('测试集上的MAE/RMSE/R2')
print(mean_absolute_error(y_test_predict, y_test))
print(np.sqrt(mean_squared_error(y_test_predict, y_test)))
print(r2_score(y_test_predict, y_test))

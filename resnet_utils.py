#resnet的支持库
import keras
from keras.layers import Dense, Dropout,Flatten,Conv1D,MaxPooling1D,BatchNormalization,AveragePooling1D
from keras.layers.core import Activation
from keras.models import Model, Input

def Unit(x,filters,pool=False):
    res = x
    if pool:
        x = MaxPooling1D(pool_size=(2))(x)
        res = Conv1D(filters=filters,kernel_size=[1],strides=(2),padding="same")(res)
    out = BatchNormalization()(x)
    out = Activation("relu")(out)
    out = Conv1D(filters=filters, kernel_size=[3], strides=[1], padding="same")(out)
 
    out = BatchNormalization()(out)
    out = Activation("relu")(out)
    out = Conv1D(filters=filters, kernel_size=[3], strides=[1], padding="same")(out)
 
    out = keras.layers.add([res,out])
 
    return out
    
def MiniModel(input_shape):
    data = Input(input_shape)
    net = Conv1D(filters=32, kernel_size=[3], strides=[1], padding="same")(data)
    net = Unit(net,32)
    net = Unit(net,32)
    net = Unit(net,32)
 
    net = Unit(net,64,pool=True)
    net = Unit(net,64)
    net = Unit(net,64)
 
    net = Unit(net,128,pool=True)
    net = Unit(net,128)
    net = Unit(net,128)
 
    net = BatchNormalization()(net)
    net = Activation("relu")(net)
    net = Dropout(0.25)(net)
 
    net = AveragePooling1D(pool_size=(4))(net)
    net = Flatten()(net)
    net = Dense(units=32,activation="sigmoid")(net)
    net = Dense(units=3,activation="softmax")(net)
    
    model = Model(inputs=data,outputs=net)
 
    return model
import numpy as np

NUM_OF_SAMPLE = 10
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import keras.backend as K
import tensorflow as tf

img = load_img('./dataset224_2/image0.jpg')
img_array = img_to_array(img)

y = np.zeros((NUM_OF_SAMPLE, *img_array.shape))
for i in range(NUM_OF_SAMPLE):
    filename = './dataset224_2/image' + str(i) + '.jpg'
    img = load_img(filename)
    img_array = img_to_array(img)
    y[i] = img_array / 255 - 0.5

features = []
with open('./dataset224_2/test0.mhm', 'r') as file:
    a = '0'
    while (a[:8] != 'modifier'):
        a = file.readline()
    while (a[:8] == 'modifier'):
        s = a.split()
        features.append(s[0] + ' ' + s[1] + ' ')
        a = file.readline()
num_features = len(features)
zero_features=[]
ones_feature=[]
for i in range(1):
    filename = './dataset224_2/test' + str(i) + '.mhm'
    with open(filename, 'r') as file:
        while (a[:8] != 'modifier'):
            a = file.readline()
        feature_idx = 0
        while (a[:8] == 'modifier'):
            s = a.split()
            if s[-2]=='macrodetails-height/Height':
                ones_feature.append(feature_idx)
            elif '|' not in s[-2]:
                zero_features.append(feature_idx)
            feature_idx += 1
            a = file.readline()
print(zero_features)
print(ones_feature)

X = np.zeros((NUM_OF_SAMPLE, num_features))
for i in range(NUM_OF_SAMPLE):
    filename = './dataset224_2/test' + str(i) + '.mhm'
    with open(filename, 'r') as file:
        while (a[:8] != 'modifier'):
            a = file.readline()
        feature_idx = 0
        while (a[:8] == 'modifier'):
            s = a.split()
            if '|' not in s[-2]:
                X[i][feature_idx] = float(s[-1]) * 2 - 1
            else:
                X[i][feature_idx] = float(s[-1])
            X[i][feature_idx] = float(s[-1])
            feature_idx += 1
            a = file.readline()

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, Reshape, UpSampling2D, BatchNormalization

from keras.layers import LeakyReLU


def make_model():
    model = Sequential()
    model.add(Dense(256 * 4 * 4, input_dim=152, use_bias=False))
    model.add(LeakyReLU(alpha=0.02))
    model.add(Reshape((4, 4, 256)))
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8, trainable=False))
    model.add(LeakyReLU(alpha=0.02))
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8, trainable=False))
    model.add(LeakyReLU(alpha=0.02))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8, trainable=False))
    model.add(LeakyReLU(alpha=0.02))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8, trainable=False))
    model.add(LeakyReLU(alpha=0.02))
    model.add(UpSampling2D())
    model.add(Conv2D(32, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8, trainable=False))
    model.add(LeakyReLU(alpha=0.02))
    model.add(UpSampling2D())
    model.add(Conv2D(3, kernel_size=4, padding="same"))
    model.add(Activation("tanh"))

    return model


model = make_model()
model.load_weights('model_224_2_00001800.hdf5')

from keras.layers import Layer


class Reshape_Layer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Reshape_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(0, 0),
                                      initializer='uniform',
                                      trainable=False)
        super(Reshape_Layer, self).build(input_shape)

    def call(self, x):
        return x[:, 16:240, 16:240, :]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 224, 224, 3)


from keras_vggface.vggface import VGGFace

model_VGG = VGGFace(model='resnet50')
model_F1 = Sequential()

model_F1.add(model)
model_F1.add(Reshape_Layer((224, 224, 3)))
model_F1.add(model_VGG)
model_true = Sequential()
model_true.add(Reshape((256, 256, 3), input_shape=(256, 256, 3)))
model_true.add(Reshape_Layer((224, 224, 3)))
model_true.add(model_VGG)

y_true = model_F1.predict(X)
y_true = y_true[1]

model_F1_predict = Sequential()
model_F1_predict.add(Dense(152, use_bias=False, input_dim=1))
model_F1_predict.add(model_F1)



def resize_to256(arr):
    arr256 = np.zeros(256)
    for i in range(0, arr.size, 256):
        arr256[i // 256] = arr[i:i + 256].sum()
    return arr256


def loss_cos(y_true, y_pred):
    if isinstance(y_pred, np.ndarray):
        a = y_true.flatten()
        b = y_pred.flatten()
        return 1 - sum(a * b) / np.sqrt(sum(a * a) * sum(b * b))
        # return ((a - b)*(a-b)).sum()
    a = K.flatten(y_true)
    b = K.flatten(y_pred)
    return 1 - K.sum(a * b) / K.sqrt(K.sum(a * a) * K.sum(b * b))
    # return K.sum((a - b)*(a-b))


def eval_loss(x):
    x = x.reshape((1, 152))
    l = []
    l.append(x)
    model_F1_predict.layers[0].set_weights(l)
    y_pred = model_F1_predict.predict([[1.]])
    loss = loss_cos(y_true, model_F1_predict.output)
    loss_value = loss_cos(y_true, y_pred)
    grad = K.gradients(loss, model_F1_predict.weights[0])
    f = K.function([model_F1_predict.input], grad)
    loss_value = tf.cast(loss_value, tf.float64)
    grad_value = (f(np.array([[1.]])))[0]
    grad_value = tf.cast(grad_value, tf.float64)
    # print(loss_value.numpy())
    # print(x[0][:10])
    return loss_value.numpy(), grad_value.numpy()


from scipy import optimize

x = np.random.randn(152) * 0.25
for i in zero_features:
    x[i]=(x[i]+1)/2
for i in ones_feature:
    x[i]=1
bounds = [(1,1) if i in ones_feature else (0,1) if i in zero_features else (-1, 1) for i in range(152)]
for i in range(1000):
    print('Start of iteration', i)
    x, min_val, info = optimize.fmin_l_bfgs_b(eval_loss, x.flatten(), bounds=bounds,
                                              maxfun=0, pgtol=1e-25,factr=10)

    print('Current loss value:', min_val)
    print('Current |grad|:', (abs(info['grad'])).sum())
    print('Sum abs dif:',(abs(x-X[1:2][0])).sum())
    print('Max abs dif:',(abs(x - X[1:2][0])).max())
    if i%1==0:
        print(x)

print(x)


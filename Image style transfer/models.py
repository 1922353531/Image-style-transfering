# 构建VGG（仅需使用预训练VGG-19中的卷积层和池化层参数）

import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc

import settings


class Model():

    def __init__(self, content_path, style_path):
        self.content = self.load_img(content_path)  # 加载内容图片
        self.style = self.load_img(style_path)  # 加载风格图片
        self.random_img = self.get_random_img()  # 生成噪音内容图片（生成一张随机图片）
        self.net = self.vggnet()  # 构建VGG网络

    def vggnet(self):
        vgg = scipy.io.loadmat(settings.VGG_MODEL_PATH)# 读取预训练的VGG-19模型
        vgg_layers = vgg['layers'][0]
        net = {}
        # 使用预训练的模型参数构建VGG网络的卷积层和池化层
        net['input'] = tf.Variable(np.zeros([1, settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH, 3]), dtype=tf.float32)

        # 参数对应的层数可以参考VGG-19模型图
        net['conv1_1'] = self.conv_relu(net['input'], self.get_wb(vgg_layers, 0))
        net['conv1_2'] = self.conv_relu(net['conv1_1'], self.get_wb(vgg_layers, 2))
        net['pool1'] = self.pool(net['conv1_2'])
        net['conv2_1'] = self.conv_relu(net['pool1'], self.get_wb(vgg_layers, 5))
        net['conv2_2'] = self.conv_relu(net['conv2_1'], self.get_wb(vgg_layers, 7))
        net['pool2'] = self.pool(net['conv2_2'])
        net['conv3_1'] = self.conv_relu(net['pool2'], self.get_wb(vgg_layers, 10))
        net['conv3_2'] = self.conv_relu(net['conv3_1'], self.get_wb(vgg_layers, 12))
        net['conv3_3'] = self.conv_relu(net['conv3_2'], self.get_wb(vgg_layers, 14))
        net['conv3_4'] = self.conv_relu(net['conv3_3'], self.get_wb(vgg_layers, 16))
        net['pool3'] = self.pool(net['conv3_4'])
        net['conv4_1'] = self.conv_relu(net['pool3'], self.get_wb(vgg_layers, 19))
        net['conv4_2'] = self.conv_relu(net['conv4_1'], self.get_wb(vgg_layers, 21))
        net['conv4_3'] = self.conv_relu(net['conv4_2'], self.get_wb(vgg_layers, 23))
        net['conv4_4'] = self.conv_relu(net['conv4_3'], self.get_wb(vgg_layers, 25))
        net['pool4'] = self.pool(net['conv4_4'])
        net['conv5_1'] = self.conv_relu(net['pool4'], self.get_wb(vgg_layers, 28))
        net['conv5_2'] = self.conv_relu(net['conv5_1'], self.get_wb(vgg_layers, 30))
        net['conv5_3'] = self.conv_relu(net['conv5_2'], self.get_wb(vgg_layers, 32))
        net['conv5_4'] = self.conv_relu(net['conv5_3'], self.get_wb(vgg_layers, 34))
        net['pool5'] = self.pool(net['conv5_4'])

        return net


    # 构建卷积与非线性函数运算，其中wb[0],wb[1] == weight,biases
    def conv_relu(self, input, wb):

        conv = tf.nn.conv2d(input, wb[0], strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(conv + wb[1])

        return relu


    # 进行最大池化运算
    def pool(self, input):

        max_pool = tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        return max_pool


    # 从预训练好的VGG-19模型中读取参数，由于VGG-19是预训练模型，其中的weights与biases均为常量
    def get_wb(self, layers, i):
    # layers为VGG-19，i为指定的VGG-19的对应层数

        w = tf.constant(layers[i][0][0][0][0][0])
        bias = layers[i][0][0][0][0][1]
        b = tf.constant(np.reshape(bias, (bias.size)))

        return w, b


    # 根据噪音和内容图片，生成一张随机图片（此时并没有设计风格图片）
    def get_random_img(self):

        noise_image = np.random.uniform(-20, 20, [1, settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH, 3])
        random_img = noise_image * settings.NOISE + self.content * (1 - settings.NOISE)

        return random_img


    # 加载一张图片，将其转化为符合要求的格式
    def load_img(self, path):

        image = scipy.misc.imread(path)# 读取图片
        image = scipy.misc.imresize(image, [settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH])# 重新设定图片大小
        image = np.reshape(image, (1, settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH, 3))# 改变数组形状，其实就是把它变成一个batch_size=1的batch
        image = image - settings.IMAGE_MEAN_VALUE # 减去均值，使其数据分布接近0（归一化操作）

        return image

if __name__ == '__main__':
    Model(settings.CONTENT_IMAGE, settings.STYLE_IMAGE)


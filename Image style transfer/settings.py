# 内容图片路径
CONTENT_IMAGE = 'images/content/Beauty.jpg'

# 风格图片路径
STYLE_IMAGE = 'images/style/style_post-impressionism.jpg'

# 输出图片路径
OUTPUT_IMAGE = 'output/Beauty/Beauty_output'

# 预训练的vgg模型路径
VGG_MODEL_PATH = 'imagenet-vgg-verydeep-19.mat'

# 图片宽度
IMAGE_WIDTH = 300

# 图片高度
IMAGE_HEIGHT = 450

# 定义计算内容损失的VGG-19层名称及对应权重的列表
CONTENT_LOSS_LAYERS = [('conv4_2', 0.5), ('conv5_2', 0.5)]

# 定义计算风格损失的VGG-19层名称及对应权重的列表
STYLE_LOSS_LAYERS = [('conv1_1', 0.2), ('conv2_1', 0.2), ('conv3_1', 0.2), ('conv4_1', 0.2), ('conv5_1', 0.2)]

# 噪音比率（用于生成一张随机噪声图片，根据风格和内容loss来不断调整其）
NOISE = 0.5

# 图片RGB均值（将像素矩阵中的所有元素归一化）
IMAGE_MEAN_VALUE = [128.0, 128.0, 128.0]

# 内容损失权重
ALPHA = 1

# 风格损失权重
BETA = 500

# 训练次数
TRAIN_STEPS = 1000
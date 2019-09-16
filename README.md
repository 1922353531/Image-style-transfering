# Image style transfer
## Abstract
This projects uses CNN model to implement image style transfering with Tensorflow.
## How to get start
### Structure
1.images/content contains the images which you want to change style.  
2.images/style contains the images which you want to get style.  
3.outputs will contain the generated images which have content of image in image/content and style in image/style.    
3.model.py contains the retrained-CNN model's weights in order to generate the final image more quickly.  
4.settings.py contains some hyper-parameters of the network.  
5.training.py allows you train the network by yourself and the generated images will be saved in images/outputs automatically.
### Training by yourself
After filling in the paths of content image and style image, you just run the training.py is OK.
## PS
The retrained weights and model's structure should be downloaded by yourself or you can say it in Issues and give me your email so I can send it to you.

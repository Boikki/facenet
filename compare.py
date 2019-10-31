# -*- coding:utf-8 -*-
#用来计算两张人脸图像之间的距离矩阵。需要输入的参数：
#预训练模型 图片1  图片220170512-110547 1.png 2.png

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import facenet
import cv2
import align.detect_face
from PIL import Image
import matplotlib.pyplot as plt


def main(args):

    images = load_and_align_data(args.image_files, args.image_size, args.margin, args.gpu_memory_fraction)
    # plt.figure()
    # plt.imshow(images[1,:])
    # plt.show()
    # print('askhnauisd')

    with tf.Graph().as_default():

        with tf.Session() as sess:
      
            # Load the model
            facenet.load_model(args.model)
    
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
            
            nrof_images = len(args.image_files)

            print('Images:')
            for i in range(nrof_images):
                print('%1d: %s' % (i, args.image_files[i]))
            print('')
            
            # Print distance matrix
            print('Distance matrix')
            print('    ', end='')
            for i in range(nrof_images):
                print('    %1d     ' % i, end='')
            print('')
            for i in range(nrof_images):
                print('%1d  ' % i, end='')
                for j in range(nrof_images):
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
                    print('  %1.4f  ' % dist, end='')
                print('')


# image_paths
# image_size
# margin
# gpu_memory_fraction

img_temp = 1

def load_and_align_data(image_paths, image_size=160, margin=44, gpu_memory_fraction=1.0):
    
    global img_temp

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
            # pnet, rnet, net = detect_face.create_mtcnn(sess, None)

    tmp_image_paths = image_paths.copy()
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        # bounding_boxes: 左上角x，y，右下角x，y，可信度
        bounding_boxes = [int(x) for x in bounding_boxes[0]]
        x1, y1, x2, y2, _ = bounding_boxes
        print("bounding_boxes: ", bounding_boxes)
        print("img size: ", img.shape)
        cropped = img[y1:y2, x1:x2]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        #cropped = img
        #aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        #prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
        #cv2.imwrite(str(img_temp)+'.jpg', aligned)
        #img_temp += 1
    images = np.stack(img_list)
    return images
    # return img

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('image_files', type=str, nargs='+', help='Images to compare')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    #这是一个从外部输入参数的代码。
    main(parse_arguments(sys.argv[1:]))

#!/usr/bin/env python
# coding=utf-8

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import facenet
import cv2
import align.detect_face
from PIL import Image

class Detector:
    def __init__(self):
        self.image_size = 160
        self.margin = 44
        self.gpu_memory_fraction = 1.0
        self.minsize = 20
        self.threshold = [ 0.6, 0.7, 0.7 ]
        self.factor = 0.709
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(sess, None)

    def get_face_by_file(self, file_path):
        img = misc.imread(file_path, mode='RGB')
        bounding_boxes, _ = align.detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        if bounding_boxes.shape[0] == 0:
            return [None, None]
        bounding_boxes = [int(x) for x in bounding_boxes[0]]
        x1, y1, x2, y2, _ = bounding_boxes
        cropped = img[y1:y2, x1:x2]
        aligned = misc.imresize(cropped, (self.image_size, self.image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        return [prewhitened, aligned]

    def get_face_by_array(self, img):
        bounding_boxes, _ = align.detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        if bounding_boxes.shape[0] == 0:
            return [None, None]
        bounding_boxes = [int(x) for x in bounding_boxes[0]]
        x1, y1, x2, y2, _ = bounding_boxes
        cropped = img[y1:y2, x1:x2]
        aligned = misc.imresize(cropped, (self.image_size, self.image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        return [prewhitened, aligned]

if __name__ == '__main__':
    d = Detector()
    print(d.get_face_by_file('./faces/xuhaoran.jpg'))
    img = cv2.imread('./faces/xuhaoran2.jpg')
    print(d.get_face_by_array(img))

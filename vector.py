import tensorflow as tf
import numpy as np
import sys
import os
import facenet
import time
import cv2
from apscheduler.schedulers.background import BackgroundScheduler


class Vector:
    def __init__(self, model='20190218-164145'):
        with tf.Graph().as_default():
            with tf.Session().as_default() as self.sess:
                facenet.load_model(model)
                self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        if os.path.exists('faces.npy'):
            self.faces_known = np.load('faces.npy', allow_pickle=True).tolist()
        else:
            self.faces_known = []
    
    def check(self, face_img):
        v = self.output(face_img)
        min_distant = 1 << 10
        rst = None
        for i in self.faces_known:
            dis = self.distant(v, i[1])
            print(i[0], dis)
            if dis < 0.7 and dis < min_distant:
                min_distant = dis
                rst = i[0]
        return rst

    def cos_similar(self, v1, v2):
        upper = 0
        for i in range(len(v1)):
            upper += v1[i] * v2[i]
        temp1 = 0
        temp2 = 0
        for i in v1:
            temp1 += i ** 2
        for i in v2:
            temp2 += i ** 2
        lower = np.sqrt(temp1) * np.sqrt(temp2)
        return upper / lower
    
    def distant(self, v1, v2):
        v1 = np.array(v1)
        v2 = np.array(v2)
        return np.sqrt(np.sum(np.square(np.subtract(v1, v2))))

    def output(self, image):
        temp = [image]
        image = np.stack(temp)
        feed_dict = { self.images_placeholder: image, self.phase_train_placeholder: False}
        emb = self.sess.run(self.embeddings, feed_dict=feed_dict)
        return emb[0]
        
if __name__ == '__main__':
    t1 = time.time()
    v = Vector()
    print(time.time() - t1)
    img = cv2.imread('./faces/xuhaoran2.jpg')
    img = cv2.resize(img, (160, 160))
    print(v.check(img))



#!/usr/bin/env python
# coding=utf-8

from camera import Camera
from vector import Vector
import cv2
#import open
import time

def open_door(input):
    print('*'*10)
    print('OPEN!')
    print('*'*10)
    #open.open()
    print(input)

def main():
    V = Vector()
    C = Camera()
    C.start()
    while True:
        img, _ = C.get_image()
        if img is None:
            time.sleep(0.5)
            continue
        cv2.imshow('1', _)
        cv2.waitKey(1000)
        rst = V.check(img)
        if rst is None:
            continue
        open_door(rst)
        C.pause()
        C.clear()
        time.sleep(3)
        C.resume()

if __name__ == '__main__':
    main()

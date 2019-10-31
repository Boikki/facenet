#!/usr/bin/env python
# coding=utf-8

import numpy as np
import cv2
import os
import sys
from detector import Detector

if os.path.exists('faces.npy'):
    faces_known = np.load('faces.npy', allow_pickle=True).tolist()
else:
    faces_known = []

if __name__ == '__main__':
    help = '''
        usage:
        python managment.py [option] [path_or_name]
        [option]:
        - [add] picture_dic_path
            it will make a face picture in ./temp, you can check it if the picture has a faces.npy
        - [del] name
            delete the person from datebase. Well, althogh the project does not have a db.
        - [del] --all
            delete all person
        - [show]
            show all name in db
        '''
    if len(sys.argv) < 2:    
        print(help)
    if sys.argv[1] == 'add':
        from vector import Vector
        V = Vector()
        D = Detector()
        files = os.listdir(sys.argv[2])
        for file in files:
            name = file.split('.')[0]
            for each in faces_known:
                if name == each[0]:
                    print("%s has in the db"%name)
                    continue
            img = cv2.imread(sys.argv[2].rstrip('/') + '/' + file)
            img, _ = D.get_face_by_array(img)
            cv2.imwrite('./temp/'+name+'.jpg', _)
            vec = V.output(img)
            faces_known.append([name, vec])
            np.save('faces.npy', faces_known)
            print("%s was added in the db successfully"%name)
    elif sys.argv[1] == 'del':
        if sys.argv[2] == '--all':
            for i, each in enumerate(faces_known):
                faces_known.pop(i)
                try:
                    os.remove('./temp/'+name+'.jpg')
                except Exception as e:
                    print(e)
                    pass
                np.save('faces.npy', faces_known)
                print('delete %s successfully'%name)
            return None
        name = sys.argv[2]   
        for i, each in enumerate(faces_known):
            if sys.argv[2] == each[0]:
                faces_known.pop(i)
                try:
                    os.remove('./temp/'+name+'.jpg')
                except Exception as e:
                    print(e)
                    pass
                np.save('faces.npy', faces_known)
                print('delete %s successfully'%name)
                exit(0)
        print("Cannot find %s"%name)
        exit(1)
    elif sys.argv[1] == 'show':
        for i in faces_known:
            print(i[0])
    else:
        print(help)




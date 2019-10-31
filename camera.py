from queue import Queue
import time
import cv2
from detector import Detector
from apscheduler.schedulers.background import BackgroundScheduler


class Camera:
    def __init__(self, queue_len = 10):
        #初始化对象
        self.q_edit = Queue(maxsize=queue_len)
        self.q_show = Queue(maxsize=queue_len)
        self.camera = cv2.VideoCapture(0)
        self.classifier = Detector()
        self.sch = BackgroundScheduler()
        self.sch.add_job(self.work_once, 'interval', seconds=0.5)
    
    def work_once(self):
        try:
            if self.camera.isOpened():
                ok, img = self.camera.read()
                if not ok:
                    print('camera cannot be read!')
                    exit(2)
                face_edit, face_show = self.classifier.get_face_by_array(img)
                if face_edit is None:
                    return None
                if self.q_edit.full():
                    self.q_edit.get()
                    self.q_show.get()
                self.q_edit.put(face_edit)
                self.q_show.put(face_show)
            else:
                print('camera cannot open!')
                exit(3)
        except Exception as e:
            print(e)
            pass

    #让相机开始工作
    def start(self):
        self.sch.start()
    
    def pause(self):
        self.sch.pause()

    def resume(self):
        self.sch.resume()

    def end(self):
        self.sch.shutdown()
        self.camera.release()
        cv2.destroyAllWindows() 

    def get_image(self):
        #从队列弹出图片
        #如果队列为空，则返回None
        if self.q_edit.empty():
            return [None, None]
        #否则返回队首的已经处理过的面部图片的文件名
        else:
            rst = [self.q_edit.get(), self.q_show.get()]
            return rst
    
    def clear(self):
        self.q_edit.queue.clear()
        self.q_show.queue.clear()

if __name__ == '__main__':
    a = Camera()
    a.start()
    while True:
        _, face = a.get_image()
        if face is None:
            continue
        cv2.imwrite('1.jpg', face)






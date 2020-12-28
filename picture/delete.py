import os 
import shutil
import cv2 as cv  

if __name__ == "__main__":
    cv.namedWindow("img", 0)
    for i in range(1,10,):
        path = "./%d"%i
        for f in os.listdir(path):
            full_path = os.path.join(path, f)
            img = cv.imread(full_path)
            cv.imshow("img", img)
            key = cv.waitKey(0)
            if(key == ord('d')):
                drop_path = os.path.join('./drop', path)
                if not os.path.exists(drop_path):
                    os.mkdir(drop_path)
                shutil.move(full_path, drop_path)

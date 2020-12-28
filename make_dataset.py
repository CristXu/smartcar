import os 
import cv2 as cv 
import numpy as np 

make = True 
check = True
if __name__ == "__main__":
    if make:
        all_data = []
        all_label = []
        numpy_data = []
        numpy_label = []
        for i in range(10):
            path = './picture/%d'%i  
            for f in os.listdir(path):
                extension = os.path.splitext(f)[-1]
                if ( extension == '.jpg'):
                    img = cv.imread(os.path.join(path, f))
                    try:
                        img = cv.resize(img, (32,32))
                        all_data.append(img)
                        all_label.append(i)
                    except:
                        continue
                if (extension == '.npy'):
                    npy_file = os.path.join(path, f)
                    tmp = np.load(npy_file)
                    numpy_data.append(tmp)
                    numpy_label += [i] * (len(tmp))
        npy_tmp = numpy_data[0]
        for npy in numpy_data[1:]:
            npy_tmp = np.vstack([npy_tmp, npy])

        all_data = np.asarray(all_data)
        all_data = np.vstack([all_data, npy_tmp]) if len(all_data) else npy_tmp

        all_label = np.asarray(all_label + numpy_label)

        np.save("x", all_data)
        np.save("y", all_label)
    if check:
        x = np.load("x.npy")
        y = np.load("y.npy")
        label = ["猫", "狗", "马", "猪", "牛"] + ["苹果", "橘子", "香蕉", "榴莲", "葡萄"]
        for d,idx in zip(x, y):
            print("Class %s"%label[idx])
            cv.imshow("img", d)
            cv.waitKey(20)

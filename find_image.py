import urllib.request
import urllib.parse
import os
import re
import _thread 
import numpy as np

#coding=utf-8
#添加header，其中Referer是必须的,否则会返回403错误，User-Agent是必须的，这样才可以伪装成浏览器进行访问
header=\
{
     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
     "referer":"https://image.baidu.com"
    }
url = "https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord={word}&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=0&word={word}&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&cg=girl&pn={pageNum}&rn=30&gsm=1e00000000001e&1490169411926="
error= 0#错误

from keras.datasets import cifar10, cifar100

# brg 2 rgb
(x, y),(_,_) = cifar100.load_data() 
pos = np.where(np.ravel(y) == 19)
casttle = x[pos]
np.save("./picture/4/casttle.npy", casttle[:500,:,:,(2,1,0)])

pos = np.where(np.ravel(y) == 0)
apple = x[pos]
np.save("./picture/5/apple.npy", apple[:500,:,:,(2,1,0)])

(x, y),(_,_) = cifar10.load_data() 

pos = np.where(np.ravel(y) == 3)
cat = x[pos]
np.save("./picture/0/cat.npy", cat[:500,:,:,(2,1,0)])

pos = np.where(np.ravel(y) == 5)
dog = x[pos]
np.save("./picture/1/dog.npy", dog[:500,:,:,(2,1,0)])

pos = np.where(np.ravel(y) == 7)
horse = x[pos]
np.save("./picture/2/horse.npy", horse[:500,:,:,(2,1,0)])

keywords = ["猫", "狗", "马", "猪", "牛"] + ["苹果", "橘子", "香蕉", "榴莲", "葡萄"]
total_num = 600

all_done = 0
def download(name, idx, keyword):
    keyword =urllib.parse.quote(keyword,'utf-8')#文字转码
    #设置下载位置
    download_path = "./picture/%s"%(idx)
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    n = 0#页数
    j = 0#图片名字
    step = 0
    while n <3000:
        n += 1
        #获取请求
        url1=url.format(word =keyword,pageNum = str(n * step))#获取索要查询的关键字
        #获取数据
        try:
            rep  = urllib.request.Request(url1,headers=header)
            rep = urllib.request.urlopen(rep)
            html = rep.read().decode('utf-8')
        except:
            print("数据出错")
            error = 1
            print("当前页是",str(n))
            if error ==1:
                continue
        #利用正则取出图片网址
        pattern = re.compile('thumbURL":"(.*?)"')
        data =re.findall(pattern,html)
        #下载
        for i in data:
            name = download_path + "/pic{num}.jpg".format(num =j)
            try:
                urllib.request.urlretrieve(i, name)
            except:
                print("Something failed!")
                continue
            print("save %s done!"%name)
            j+=1
        step = len(data)
        if(j >= total_num):
            all_done += 1
            break

for idx, keyword in enumerate(keywords):
    _thread.start_new_thread(download, ("%s"%idx, idx, keyword))

while(1):
    if(all_done == len(keywords)):
        print("All done!")
        break
    pass
#-*- encoding:gbk -*-
import cv2
import numpy as np
import os,codecs
from sklearn.decomposition import PCA
import sklearn.preprocessing as preprocessing
import Hist
#import sklearn

from sklearn.cluster import KMeans
def just_atest():
    #src=cv.imread('venv/Lib/101_ObjectCategories/airplanes/image_0001.jpg')
    img=cv2.imread('101_ObjectCategories/airplanes/image_0001.jpg')

    inputImgPath ='101_ObjectCategories/airplanes/image_0001.jpg'
    outputImgPath = 'test_output/test.txt'

    # featureSun:计算特征点个数
    featureSum = 0
    img = cv2.imread(inputImgPath)
    #将图片resize
    cv2.imshow('origin',img)
    img =cv2.resize(img,(200,82),interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    detector = cv2.xfeatures2d.SIFT_create()
    # 找到关键点，关键点的位置存到des
    kps , des = detector.detectAndCompute(gray,None)
    # 绘制关键点
    img=cv2.drawKeypoints(gray,kps,img)

    # 将特征点保存
    np.savetxt(outputImgPath ,des ,  fmt='%.2f')
    featureSum += len(kps)
    cv2.imshow('result',img)
    cv2.waitKey(0)
    print('kps:' + str(featureSum))

    #读取一个像素
    px=img[100,100]
    print(px)
    #读取像素中的一个颜色，就是在[2]里面加一个0-2
    blue=img[100,100,0]
    print(blue)
    #窗口显示图片
    img=cv2.imshow('test', img)


def test_ke():
    features=[[0,0,0,0,0,0],[0,0,0,3,0,1],[0,2,32,0,0,0],[1,0,0,34,0,0],[0,-2,1,0,0,0]]
    input_x = np.array(features)

    kmeans = KMeans(n_clusters=3, n_init=100,init="kmeans++",random_state=None).fit(input_x)

    return kmeans.labels_, kmeans.cluster_centers_

def meanX(dataX):
    return np.mean(dataX,axis=0)


def check(data):
    label0 = [0, 0, 0]
    for f in data:
        #total+=1
        if f.find('118_')!=-1:
            label0[2] += 1
        elif f.find('69_')!=-1:
            label0[1] += 1
        elif f.find('60_')!=-1:
            label0[0] += 1
    print(label0)
    return label0


def pca(XMat, k):
    average = meanX(XMat)
    m, n = np.shape(XMat)
    data_adjust = []
    avgs = np.tile(average, (m, 1))
    data_adjust = XMat - avgs
    covX = np.cov(data_adjust.T)   #计算协方差矩阵
    featValue, featVec=  np.linalg.eig(covX)  #求解协方差矩阵的特征值和特征向量
    index = np.argsort(-featValue) #依照featValue进行从大到小排序
    finalData = []
    if k > n:
        print "k must lower than feature number"
        return
    else:
        #注意特征向量时列向量。而numpy的二维矩阵(数组)a[m][n]中，a[1]表示第1行值
        selectVec = np.matrix(featVec.T[index[:k]]) #所以这里须要进行转置
        finalData = data_adjust * selectVec.T
        reconData = (finalData * selectVec) + average
    return finalData, reconData

def get_file_name(path):
    '''
    Args: path to list;  Returns: path with filenames
    '''
    filenames = os.listdir(path)
    path_filenames = []
    filename_list = []
    for file in filenames:
        if not file.startswith('.'):
            path_filenames.append(os.path.join(path, file))
            filename_list.append(file)

    return path_filenames

def kmeansvar(file_list, cluster_nums, n_components=5, randomState=None):
    features = []
    files = file_list
    pic_count = 0
    for file in files:
        # print(file)
        img = cv2.imread(file)
        img = cv2.resize(img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_CUBIC)
        b, g, r = cv2.split(img)
        histImgR = Hist.calcAndDrawHist(r, [255, 0, 0])
        histImgG = Hist.calcAndDrawHist(g, [0, 255, 0])
        histImgB = Hist.calcAndDrawHist(b, [0, 0, 255])
        use=[]
        use.append(histImgR)
        use.append(histImgG)
        use.append(histImgB)
        colorvar=np.array(use)
        colorstd=np.std(colorvar,axis=0)     #计算各颜色分量每一列的标准差，来看峰值之间的差距
        pic_count += 1
        features.append(colorstd)
    input_x = np.array(features)
    input_x = input_x.reshape(pic_count, 256)
    pca = PCA(n_components=n_components)
    pca.fit(input_x)
    input_x = pca.transform(input_x)
    input_x.reshape(pic_count, n_components)
    kmeans = KMeans(n_clusters=cluster_nums, init="k-means++",n_init=100,random_state=randomState).fit(input_x)

    return kmeans.labels_, kmeans.cluster_centers_


def kmeansdi(file_list, cluster_nums, n_components=5, randomState=None,color=[255,255,255],single='f'):
    features = []
    files = file_list
    pic_count = 0
    for file in files:
        #print(file)
        img = cv2.imread(file)
        img = cv2.resize(img,None,fx=0.6,fy=0.6,interpolation = cv2.INTER_CUBIC)
        b, g, r = cv2.split(img)
        #histImgG = Hist.calcAndDrawHist(g, [0, 255, 0])
        if single=='f':
            histImgG = Hist.calcAndDrawHist(img, color)  #找紫色的花=红+蓝
        elif single=='g':
            histImgG=Hist.calcAndDrawHist(g, [0,255,0])
        elif single == 'r':
            histImgG = Hist.calcAndDrawHist(r, [255,0,0])
        elif single == 'b':
            histImgG = Hist.calcAndDrawHist(b, [0,0,255])

        #reshape_feature = histImgG.reshape(-1, 1)  # 把128列降到1列，顺着存进r_f
        pic_count+=1
        #sb=np.array(histImgG)
        #pca = PCA(n_components=n_components)
        #pca.fit(sb)
        #sb=pca

        #sb=np.mean(histImgG)
        features.append(histImgG)

    input_x = np.array(features)
    input_x=input_x.reshape(pic_count,256)
    pca=PCA(n_components=n_components)
    pca.fit(input_x)
    input_x=pca.transform(input_x)
    input_x.reshape(pic_count,n_components)
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(input_x)


    kmeans = KMeans(n_clusters=cluster_nums, init="k-means++",n_init=100,random_state=randomState).fit(X_train_minmax)

    return kmeans.labels_, kmeans.cluster_centers_

def check_plane(filepath=['a','b'],token=False):
    if token==False:
        path_filenames = get_file_name("101_ObjectCategories/test_graph")  #读取所有图片
    else:
        path_filenames=filepath
    labels, cluster_centers = kmeansdi(path_filenames, 3,n_components=5,color=[96,96,96],single='f')    #先通过灰色进行二分类


    #gray = np.count_nonzero(labels)  # gray多于color，1作为gray存
    gray = np.array(labels)
    key = np.unique(gray)
    result = [0,0,0]
    for k in key:
        mask = (gray == k)
        arr_new = gray[mask]
        v = arr_new.size
        result[k] = v
    sb=np.array(result)
    aa=np.argsort(sb)
    results=[]
    result_use=[]
    res = dict(zip(path_filenames, labels))
    for f, l in res.items():
        if l == aa[0]:
            result_use.append(f)
        if l == aa[2]:
            results.append(f)
    resulta=[result_use]
    resulta.append(results)
    return resulta

def cheat_duck():
    path_filenames = get_file_name("101_ObjectCategories/test_graph")  # 读取所有图片
    use=[]

    with codecs.open('label2.txt') as f1:
        for line in f1:
                a=line.find('.jpg')
                use.append([int(line[:a-5]),int(line[a-4:a])])

    with codecs.open('plane.txt') as f2:

        for line in f2:
            if line.find('plane') != -1:

                a = line.find('.jpg')
                use.append([int(line[:a - 5]), int(line[a - 4:a])])

    comp=[]
    for line in path_filenames:
        a=line.find('.jpg')
        comp.append([int(line[a-8:a-5]),int(line[a-4:a])])
    #ab=np.array(use)
    #ac=np.unique(use)
    #use=ac.tolist()
    res=[]
    for item in comp:
        if item in use:
            continue
        else:
            res.append(item)

    #print (res)
    label=[0,0,0]
    with codecs.open('duck.txt', 'w', encoding='utf-8') as fw:
        for item in res:
            fw.write(str(item[0])+"_00"+str(item[1])+".jpg\tlabel duck\t")
            if item[0]==60:
                label[0]+=1
                fw.write("right")
            if item[0] == 69:
                label[1] += 1
            if item[0] == 118:
                label[2] += 1
            fw.write("\n")
    print (label)

def plane_master():
    path_filenames = get_file_name("101_ObjectCategories/test_graph")  # 读取所有图片
    result=check_plane()
    tier1=result[0]
    tier1res=result[1]
    result=check_plane(tier1res,token=True)
    tier2=result[0]
    tier2res=result[1]
    result=check_plane(tier2res,token=True)
    tier3=result[0]

    #check(tier1)
    #check(tier2)

    #check(tier3)
    final_res=tier1+tier2+tier3
    #check(final_res)
    right=0
    label0=[0,0,0]
    with codecs.open("plane.txt", 'w', encoding='utf-8') as fw:
        for f in path_filenames:
            #total+=1
            if f.find('118_')!=-1:
                fw.write("{}\tlabel ".format(f[f.find('118_'):]))
                if f in final_res:
                    label0[2] += 1
                    fw.write("plane\n")
                else:
                    fw.write("None\n")
            elif f.find('69_')!=-1:
                fw.write("{}\tlabel ".format(f[f.find('69_'):]))
                if f in final_res:
                    label0[1] += 1
                    right+=1
                    fw.write("plane\tright\n")
                else:
                    fw.write("None\n")
            elif f.find('60_')!=-1:
                fw.write("{}\tlabel ".format(f[f.find('60_'):]))
                if f in final_res:
                    label0[0] += 1
                    fw.write("plane\n")
                else:
                    fw.write("None\n")
        #print(label0)
        total=label0[0]+label0[1]+label0[2]
        perc=float(right)/float(total)
        print("totally {} graph detected, {} right, correct percentage {}\n".format(total, right, perc))
        fw.write("totally {} graph detected, {} right, correct percentage {}\n".format(total, right, perc))





def knn_detect(file_list, cluster_nums,n_components=7, randomState=None):
    features = []
    files = file_list
    sift = cv2.xfeatures2d.SIFT_create()
    remove_list=[]
    #sift = cv2.SIFT()
    pic_count=0
    for file in files:
        print(file)
        img = cv2.imread(file)
        img = cv2.resize(img, (300, 150), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        print(gray.dtype)
        _, des = sift.detectAndCompute(gray, None)

        if des is None:
            remove_list.append(file)
            pic_count-=1
            continue
        pic_count+=1
        test = np.array(des)
        # X = np.array([1,1,-1,5])
        test = np.transpose(test)
        try:
            pca = PCA(n_components=n_components)
            pca.fit(test)
        except:
            remove_list.append(file)
            pic_count-=1
            continue
        else:
            sb = pca.transform(test)
            sb=np.transpose(sb)
            try:
                pca = PCA(n_components=5)
                pca.fit(sb)
            except:
                remove_list.append(file)
                pic_count -= 1
                continue
            else:
                test = pca.transform(sb)
                test = np.transpose(test)

        reshape_feature = test.reshape(-1, 1)   #把128列降到1列，顺着存进r_f

        features.append(reshape_feature.tolist())


    input_x = np.array(features)
    c=input_x.ravel()
    b=c.reshape(pic_count,35)
    #finalData, reconMat=pca(input_x,5)
    for item in remove_list:
        file_list.remove(item)
    kmeans = KMeans(n_clusters=cluster_nums,init="k-means++",n_init=100, random_state=randomState).fit(b)

    return kmeans.labels_


def res_fit(filenames, labels):
    files1=[]
    files2=[]
    labels1=[]
    labels2=[]
    files = [file.split('/')[-1] for file in filenames]
    """
    for file in filenames:
        if labels==0:
            files1.append(file.split('/')[:-12])
            labels1.append('0')
        if labels==1:
            files2.append(file.split('/')[:-12])
            labels2.append('1')
    """
    return dict(zip(files, labels))


def save(path, filename, data):
    file = os.path.join(path, filename)
    label0=[[0,0,0],[0,0,0],[0,0,0]]
    #label1=[0,0,0]
    #label2=[0,0,0]

    with codecs.open(file, 'w', encoding='utf-8') as fw:
        for f, l in data.items():

            if f.find('118_')!=-1:
                label0[l][2]+=1
                fw.write("{}\tright\n".format(f[f.find('118_'):]))

            elif f.find('69_')!=-1:
                label0[l][1]+=1
                fw.write("{}\t\n".format(f[f.find('69_'):]))

            elif f.find('60_')!=-1:
                label0[l][0]+=1
                fw.write("{}\t\n".format(f[f.find('60_'):]))

            #fw.write('\n')
    print(label0)
    #print('determine flowers,get'+label0[0][2]+' flowers out of '+(label0[0][2]+label0[1][2]))

def pcaa():
    img = cv2.imread('101_ObjectCategories/069.fighter-jet/069_0070.jpg')
    img = cv2.resize(img, (300, 150), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()

    print(gray.dtype)
    _, des = sift.detectAndCompute(gray, None)
    test=np.array(des)
    #X = np.array([1,1,-1,5])
    test=np.transpose(test)
   # try:
    #    pca = PCA(n_components=10)
     #   pca.fit(test)
    sb=pca.transform(test)
    sb=np.transpose(sb)
    print(pca.transform(test))



def check_flower_purple(filepath=['a','b'],token=False):
    if token==False:
        path_filenames = get_file_name("101_ObjectCategories/test_graph")  #读取所有图片
    else:
        path_filenames=filepath
    labels, cluster_centers = kmeansdi(path_filenames, 3,n_components=5,color=[128,0,128],single='f')    #先通过灰色进行二分类


    #gray = np.count_nonzero(labels)  # gray多于color，1作为gray存
    gray = np.array(labels)
    key = np.unique(gray)
    result = [0,0,0]
    for k in key:
        mask = (gray == k)
        arr_new = gray[mask]
        v = arr_new.size
        result[k] = v
    sb=np.array(result)
    aa=np.argsort(sb)
    res_dict2=[]
    res_goodtogo=[]
    res = dict(zip(path_filenames, labels))
    for f, l in res.items():
        if l == aa[2]:
            res_dict2.append(f)
        elif l == aa[1]:
            res_goodtogo.append(f)
    #labels, cluster_centers = kmeansdi(res_dict2, 2, single='g')
    #res_g_none = dict(zip(gray_file, labels))
    #res_dictgreen = res_fit(res_dict2, labels)
    #save('./', 'dirt.txt', res_dictdirt)
    #exclu_res = []

    #path_filenames = [file.split('/')[-1] for file in path_filenames]

    res_dict1 = res_fit(path_filenames, labels)
    save('./', 'gray.txt', res_dict1)
    #print(cluster_centers)

    return res_dict2,res_goodtogo

def main():
    path_filenames = get_file_name("101_ObjectCategories/test_graph")

    labels= knn_detect(path_filenames, 3)
    path_filenames = [file.split('/')[-1] for file in path_filenames]

    res_dict1 = res_fit(path_filenames, labels)
    save('./',  'label2.txt', res_dict1)
    #save('./',  'label2.txt', res_dict2)
def check_flower_final():
    aa,bb=check_flower_purple()
    new_file_list=[]
    labels, cluster_centers = kmeansdi(aa, 3, n_components=5, color=(0,255,0),single='f')  # 先通过灰色进行二分类
    res_dict1 = res_fit(aa, labels)
    save('./', 'label2.txt', res_dict1)
    # gray = np.count_nonzero(labels)  # gray多于color，1作为gray存
    gray = np.array(labels)
    key = np.unique(gray)
    result = [0, 0, 0]
    for k in key:
        mask = (gray == k)
        arr_new = gray[mask]
        v = arr_new.size
        result[k] = v
    sb = np.array(result)
    rank = np.argsort(sb)
    res_dict2 = []
    res_goodtogo = []
    res = dict(zip(aa, labels))
    for f, l in res.items():
        if l == rank[1]:
            res_dict2.append(f)

    for item in bb:
        res_dict2.append(item)

    flabels=[0 for item in res_dict2]
    res_dict1 = res_fit(res_dict2, flabels)
    save('./', 'label2.txt', res_dict1)


def check_duck(filepath=['a','b'],token=False):
    if token == False:
        path_filenames = get_file_name("101_ObjectCategories/test_graph")  # 读取所有图片
    else:
        path_filenames = filepath
    labels,cluster_centers=kmeansvar(path_filenames, 3, n_components=8)
    #labels, cluster_centers = kmeansdi(path_filenames, 3, n_components=5, color=[128 , 0,128], single='f')  # 先通过灰色进行二分类

    # gray = np.count_nonzero(labels)  # gray多于color，1作为gray存
    gray = np.array(labels)
    key = np.unique(gray)
    result = [0, 0, 0]
    for k in key:
        mask = (gray == k)
        arr_new = gray[mask]
        v = arr_new.size
        result[k] = v
    sb = np.array(result)
    aa = np.argsort(sb)
    res_dict2 = []
    res_goodtogo = []
    res = dict(zip(path_filenames, labels))
    for f, l in res.items():
        if l == aa[2]:
            res_dict2.append(f)
        elif l == aa[1]:
            res_goodtogo.append(f)
    # labels, cluster_centers = kmeansdi(res_dict2, 2, single='g')
    # res_g_none = dict(zip(gray_file, labels))
    # res_dictgreen = res_fit(res_dict2, labels)
    # save('./', 'dirt.txt', res_dictdirt)
    # exclu_res = []

    # path_filenames = [file.split('/')[-1] for file in path_filenames]

    res_dict1 = res_fit(path_filenames, labels)
    save('./', 'gray.txt', res_dict1)
    # print(cluster_centers)

    return res_dict2, res_goodtogo


#plane_master()
#check_flower_final()
#check_duck()
cheat_duck()
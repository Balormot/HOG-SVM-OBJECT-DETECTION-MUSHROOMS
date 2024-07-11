import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from skimage.feature import hog
from skimage.color import rgb2gray
from DBSCAN import DBSCAN

mage = mpimg.imread(f'16.jpg')
plt.imshow(mage)
plt.show()


def distance2(x, y):
    #d = np.sqrt(((x - y) ** 2).sum()) #euclid
    #d = np.max(np.abs(x-y)) # cheb
    d =  np.sum((np.abs(x-y))) # manh
    return d

def distance_base(x, y):
    d = np.sqrt(((x - y) ** 2).sum())
    return d

def perecolor(img):
    whidith = np.shape(img)[0]
    length = np.shape(img)[1]
    nw = np.copy(img)
    for i in range(whidith):
        for j in range(length):
            if (i==0 or i==(whidith-1)) or (j==0 or j==(length-1) or i==1 or i==(whidith-2)) or (j==1 or j==(length-2) or i==2 or i==(whidith-3)) or (j==2 or j==(length-3)):
                nw[i,j] = [0,0,255]
    return nw

def lengst (clst, mane):
    bg = 0
    for i in clst:
        if distance_base(i,mane)>bg:
            bg = distance_base(i,mane)
    return bg


def RBF(X, Y, gamma):
    if gamma == None:
        gamma = 1.0 / X.shape[1]

    # RBF
    K = np.exp(-gamma * np.sum((X - Y) ** 2, axis=-1))

    return K

def dfk(coef,Xex,vectors,gam,bet):
    k = np.sign(np.sum(coef*RBF(Xex,vectors,gam)) + bet)
    return k

ethalon = [200,40,40]
ethalon1 = [160,90,80]
ethalon2 = [190,140,130]
ethalon3 = [240,120,12]
ethalon4 = [229,149,98]
ethalon5 = [97,55,73]
ethalon6 = [214,137,149]
ethalon7 = [234,180,101]
ethalon8 = [196,139,65]
ethalon9 = [116,73,41]


age = mage.copy()
coor = []
for  i in range (np.shape(age)[0]):
    for j in range (np.shape(age)[1]):
        if distance2(age[i][j],ethalon) < 80 or distance2(age[i][j],ethalon1) < 30 or distance2(age[i][j],ethalon2) < 30 or distance2(age[i][j],ethalon3) < 35 or distance2(age[i][j],ethalon4) < 20 or distance2(age[i][j],ethalon5) < 20 or distance2(age[i][j],ethalon6) < 20 or distance2(age[i][j],ethalon7) < 40 or distance2(age[i][j],ethalon7) < 40 or distance2(age[i][j],ethalon9) < 20:
            age[i][j] = [0,0,255]
            coor.append([i,j])
coor = np.array(coor)

dbscan = DBSCAN(epsilon=10, min_pts=20)
clusters, noise = dbscan.fit(coor)

means = []
for i in clusters:
    means.append(np.mean(np.array(coor[i]),axis=0))
means = np.int32(means)

sd = []
for i, ft in enumerate(clusters):
    sd.append(lengst(coor[ft],means[i])/10)

gag = []
ga = mage.copy()
a = 10 # ширина
b = 50# длинна вниз
b2 =10 # длина вверх
for i,j in enumerate(means):
    #ga[j[0]-a*np.int32(np.ceil(sd[i])):j[0]+a*np.int32(np.ceil(sd[i])),j[1]-a*np.int32(np.ceil(sd[i])):j[1]+a*np.int32(np.ceil(sd[i]))] = perecolor(mage[j[0]-a*np.int32(np.ceil(sd[i])):j[0]+a*np.int32(np.ceil(sd[i])),j[1]-a*np.int32(np.ceil(sd[i])):j[1]+a*np.int32(np.ceil(sd[i]))])
    ga[j[0]-a*2:j[0]+a*np.int32(np.ceil(sd[i]))+20,j[1]-a*np.int32(np.ceil(sd[i])):j[1]+a*np.int32(np.ceil(sd[i]))] = perecolor(mage[j[0]-a*2:j[0]+a*np.int32(np.ceil(sd[i]))+20,j[1]-a*np.int32(np.ceil(sd[i])):j[1]+a*np.int32(np.ceil(sd[i]))])
    gag.append(mage[j[0]-a*2:j[0]+a*np.int32(np.ceil(sd[i])),j[1]-a*np.int32(np.ceil(sd[i])):j[1]+a*np.int32(np.ceil(sd[i]))])
plt.imshow(ga)
plt.show()


gamma_r = 0.01
#beta_r = np.array([-0.92401939])
#beta_r = np.array([-0.58926533])
beta_r = np.array([-2.15165203])
dual_coef_r = np.loadtxt('model.dual_coef_r.txt')
support_r = np.loadtxt('support_vectors_r.txt')

gamma_b = 0.05
#beta_b = np.array([-0.3888666])
#beta_b = np.array([-0.45469997])
beta_b = np.array([-0.48338117])
dual_coef_b = np.loadtxt('model.dual_coef_b.txt')
support_b = np.loadtxt('support_vectors_b.txt')

gamma_pod = 0.05
#beta_b = np.array([-0.30397187])
#beta_pod = np.array([-0.517916038])
beta_pod = np.array([-0.35397187])
dual_coef_pod = np.loadtxt('model.dual_coef_pod.txt')
support_pod = np.loadtxt('support_vectors_pod.txt')

gamma_sm = 0.05
#beta_b = np.array([-0.39168536])
#beta_sm = np.array([-0.34198983])
beta_sm = np.array([-0.38072294])
dual_coef_sm = np.loadtxt('model.dual_coef_smorch.txt')
support_sm = np.loadtxt('support_vectors_smorch.txt')

gag = []
ga = mage.copy()
a = 10 # ширина
b = 50# длинна вниз
b2 =10 # длина вверх
for i,j in enumerate(means):
    #dfk(dual_coef,X_test[5],support,gamma,b)
    #dfk(dual_coef_r,test,support_r,gamma_r,beta_r) == 1
    t = mage[j[0]-a*2:j[0]+a*np.int32(np.ceil(sd[i]))+20,j[1]-a*np.int32(np.ceil(sd[i])):j[1]+a*np.int32(np.ceil(sd[i]))]
    if t.shape[0] != 0 and t.shape[1] != 0:
        test = hog(rgb2gray(resize(mage[j[0]-a*2:j[0]+a*np.int32(np.ceil(sd[i]))+20,j[1]-a*np.int32(np.ceil(sd[i])):j[1]+a*np.int32(np.ceil(sd[i]))],(128,64))), orientations=8,pixels_per_cell=(8, 8),cells_per_block=(1, 1)) #channel_axis=-1)
        if dfk(dual_coef_sm,test,support_sm,gamma_sm,beta_sm) == 1 or dfk(dual_coef_pod,test,support_pod,gamma_pod,beta_pod) == 1 or dfk(dual_coef_b,test,support_b,gamma_b,beta_b) == 1 or dfk(dual_coef_r,test,support_r,gamma_r,beta_r) == 1:
            #ga[j[0]-a*np.int32(np.ceil(sd[i])):j[0]+a*np.int32(np.ceil(sd[i])),j[1]-a*np.int32(np.ceil(sd[i])):j[1]+a*np.int32(np.ceil(sd[i]))] = perecolor(mage[j[0]-a*np.int32(np.ceil(sd[i])):j[0]+a*np.int32(np.ceil(sd[i])),j[1]-a*np.int32(np.ceil(sd[i])):j[1]+a*np.int32(np.ceil(sd[i]))])
            ga[j[0]-a*2:j[0]+a*np.int32(np.ceil(sd[i]))+20,j[1]-a*np.int32(np.ceil(sd[i])):j[1]+a*np.int32(np.ceil(sd[i]))] = perecolor(mage[j[0]-a*2:j[0]+a*np.int32(np.ceil(sd[i]))+20,j[1]-a*np.int32(np.ceil(sd[i])):j[1]+a*np.int32(np.ceil(sd[i]))])
        gag.append(mage[j[0]-a*2:j[0]+a*np.int32(np.ceil(sd[i])),j[1]-a*np.int32(np.ceil(sd[i])):j[1]+a*np.int32(np.ceil(sd[i]))])

plt.imshow(ga)
plt.show()
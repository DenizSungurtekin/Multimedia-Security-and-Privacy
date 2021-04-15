import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage
from skimage.measure import block_reduce
import glob # Used to read all image in a directory

# 1 Introduction

peackock = cv2.imread("peacock.jpg") #Read image with colors

height,width,channels = peackock.shape
print("The image has height: ",height)
print("The image has width: ",width)
print("The image has channels: ",channels)


blueHist = cv2.calcHist([peackock],[0],None,[256],[0,256]) #arg = ([image],[0] or [1] or [2] // grey/blue-green-red, mask (hist of a region), [bin count], ranges = [0,256]
greenHist = cv2.calcHist([peackock],[1],None,[256],[0,256])
redHist = cv2.calcHist([peackock],[2],None,[256],[0,256])


window1 = 'Peackock Color'
cv2.imshow(window1,peackock) # Show image
cv2.waitKey(0)


fig = plt.figure(1)
plt.title("Blue Histogramme")
plt.plot(blueHist,color="blue")
plt.show()

fig = plt.figure(2)
plt.title("Green Histogramme")
plt.plot(greenHist,color="green")
plt.show()

fig = plt.figure(3)
plt.title("Red Histogramme")
plt.plot(redHist,color="red")
plt.show()


peackock = cv2.imread("peacock.jpg",0) #Read image without colors (Grayscale)
window2 = 'Peackock Grey'
cv2.imshow(window2,peackock) # Show image
cv2.waitKey(0)

# Global mean
globalMean = peackock.mean()
globalVariance = peackock.var()
print("Global mean: ",globalMean)
print("Global Variance: ",globalVariance)


# Local mean and local variance

localMean = skimage.measure.block_reduce(peackock,block_size=(3,3), func = np.mean)
localMean = localMean.astype(np.uint8)
window3 = 'Local mean'
cv2.imshow(window3,localMean) # Show image
cv2.waitKey(0)

localVariance = skimage.measure.block_reduce(peackock,block_size=(3,3), func = np.var)
localVariance = localVariance.astype(np.uint8)
window4 = 'Local Variance'
cv2.imshow(window4,localVariance) # Show image
cv2.waitKey(0)


# 2 Noise

def generateGaussianNoise(mean,std,N,M): #Generate a Gaussian Noise array NxM
    return np.asarray([np.random.normal(mean,std,M) for i in range(N)])


def generateSaltPeper(X,p,q): #Generate a Salt and peper Noised image for an image X
    if p+q < 1:
        s_min = np.min(X)
        s_max = np.max(X)
        N,M = X.shape
        for i in range(N):
            for j in range(M):
                r = np.random.rand()

                if r <= p:
                    X[i][j] = s_min

                if p+q > r > p:
                    X[i][j] = s_max
        return X
    else:
        print("p + q need to be smaller than 1")


def meanSquaredError(X,Y): #MSE between two images
    if X.shape == Y.shape:
        X = skimage.img_as_float(X) #need to convert the image in double format before computing the mse (To have the result in a chosen datatype)
        Y = skimage.img_as_float(Y)
        N,M = X.shape
        squaredErrorMatrix = np.asarray([[(Y[i][j] - X[i][j])**2 for j in range(M)] for i in range(N)])
        tot = N*M
        somme = sum(sum(squaredErrorMatrix))
        return somme/tot
    else:
        print("X and Y must be in same size")
#
peackockUint8 = cv2.imread("peacock.jpg",0) # uint8 by default
peackockDouble = skimage.img_as_float(peackockUint8) # convert to double format

###
print("MSE between the two dataType: ",meanSquaredError(peackockUint8,peackockDouble)) # Explanation: the mean square error compute the mean of the sum squared difference between each pixel's value so its logic to have a very big value. The uint8 datatype is between 0 and 255 and for the double we have value between 0 and 1 for each pixels.
# In addition when we add two uint8 values together the result is also in uint8 so its important to be careful.
###

# Peak Signal to Noise Ration

def PSNR(X,Y,sigma):
    if meanSquaredError(X,Y) == 0:
        return "same image"

    a = 1
    return 10 * np.log10((np.power(a,2))/(np.power(sigma,2)))

def PSNR2(X,Y):
    if meanSquaredError(X,Y) == 0:
        return "same image"

    a = 1
    return 10 * np.log10((np.power(a,2))/meanSquaredError(X,Y))



def addGaussianNoiseAndDisplayPSNR(image,sigma):
    N,M = image.shape

    image = skimage.img_as_float(image)
    window4 = 'Original Image'
    cv2.imshow(window4, image)  # Show image
    cv2.waitKey(0)

    noise = generateGaussianNoise(0,sigma, N, M)
    noisedImage = image + noise
    noisedImage = skimage.img_as_float(noisedImage)

    window5 = 'Noised Image'
    cv2.imshow(window5, noisedImage)  # Show image
    cv2.waitKey(0)

    noisedImage = np.asarray([ [round(noisedImage[i][j]*255) for j in range(M)] for i in range(N) ])
    noisedImage = noisedImage.astype(np.uint8)

    hist = cv2.calcHist([noisedImage], [0], None, [256], [0, 256])
    plt.title("Histogramme")
    plt.plot(hist,color="blue")
    plt.show()


    psnr = PSNR(image,noisedImage,sigma)
    print("Noise with sigma = ",sigma," and PNSR = ",psnr," MSE = ",meanSquaredError(image,noisedImage))



#Gaussian

image = cv2.imread("dct_db/1_IMG_8059.tif",0)
# addGaussianNoiseAndDisplayPSNR(image,0.03)


#Peper noise with psnr = 40
image = skimage.img_as_float(image)
p = 0.0001
q = 0.0002
noisedImage = generateSaltPeper(image,p,q)
image = cv2.imread("dct_db/1_IMG_8059.tif",0)
image = skimage.img_as_float(image)
sigma = noisedImage.std()
print()

psnr = PSNR2(image,noisedImage)
window6 = 'Peper Noised Image'
cv2.imshow(window6, noisedImage)  # Show image
cv2.waitKey(0)
print("Noise with p = ",p ,"q = ",q ,"and PNSR = ",psnr," MSE = ",meanSquaredError(image,noisedImage))



# 3 Identification


def hash(imagePath):
    image = cv2.imread(imagePath, 0) #Read and convert to grayscale
    globalMean = image.mean()
    localMean = skimage.measure.block_reduce(image,block_size=(32,32), func = np.mean)
    localMean = localMean.flatten()
    N = len(localMean)
    booleanVector = [localMean[i]>globalMean for i in range(N)]
    hashVector = [1 if booleanVector[i] else 0 for i in range(N)]
    return np.asarray(hashVector)

def hashNoised(image):
    globalMean = image.mean()
    localMean = skimage.measure.block_reduce(image,block_size=(32,32), func = np.mean)
    localMean = localMean.flatten()
    N = len(localMean)
    booleanVector = [localMean[i]>globalMean for i in range(N)]
    hashVector = [1 if booleanVector[i] else 0 for i in range(N)]
    return np.asarray(hashVector)


def hammingDistance(binaryVector1,binaryVector2):
    N = max(binaryVector1.shape)
    booleanVector = [binaryVector1[i] == binaryVector2[i] for i in range(N)]
    h = booleanVector.count(False)
    P = h/N
    return h,P

def read200images():
    path = "./" + "dct_db" + "/*.tif"
    imagesPaths = glob.glob(path)
    hashVectors = np.asarray([hash(imagePath) for imagePath in imagesPaths])
    return hashVectors

def read200Noisedimages(images):
    hashVectors = np.asarray([hashNoised(image) for image in images])
    return hashVectors

def write200Noisedimages(sigma):
    path = "./" + "dct_db" + "/*.tif"
    imagesPaths = glob.glob(path)
    noisedImages = []
    for path in imagesPaths:
        image = skimage.img_as_float(cv2.imread(path, 0))
        N,M = image.shape
        noise = generateGaussianNoise(0,sigma,N,M)
        noisedImage = image + noise
        noisedImages.append(noisedImage)
    return noisedImages


L = 64
X = read200images().transpose() # matrix of size LxM, L = 64, M = 200
Y = read200Noisedimages(write200Noisedimages(0.017)).transpose()  #sigma = 0.017 -> PSNR = 35db

X = np.where(X==0,-1,X)
Y = np.where(Y==0,-1,Y)

Z = np.matmul(X.transpose(),Y)
Zprime = np.matmul(X.transpose(),X)

Z = np.divide((Z - L),-2)
Zprime = np.divide((Zprime - L),-2)  # Z = Zprime


def pairProbabilityErrorSameClass(X,Y):
    X = X.transpose() #To take first element as the first vector
    Y = Y.transpose()
    N = len(X)
    probabilityErrors = [hammingDistance(X[i],Y[i])[1] for i in range(N)]
    return probabilityErrors


def pairProbabilityErrorDifferentClass(X,Y):
    X = X.transpose() #To take first element as the first vector
    Y = Y.transpose()
    N = len(X)
    probabilityErrors = [hammingDistance(X[i], Y[j])[1] for j in range(N) for i in range(N) if i != j]
    return probabilityErrors

# Compute probability error of the two different configuration

probabilityDifferentClass = pairProbabilityErrorDifferentClass(X,Y)
probabilitySameClass = pairProbabilityErrorSameClass(X,Y)

# Normalization and plot of the density histogram
# We take the sum of the element and divide each element by it, like this we know that the sum is of 1
# The histogram density with mathplotlib and numpy is a litle bit buggy so i made mine represented by a function

sumationDifferentClass = sum(probabilityDifferentClass)
sumationSameClass = sum(probabilitySameClass)

probabilityDifferentClass = [element/sumationDifferentClass for element in probabilityDifferentClass]
probabilitySameClass = [element/sumationSameClass for element in probabilitySameClass]

x = np.unique(np.asarray(probabilityDifferentClass))
y = [probabilityDifferentClass.count(element) for element in x]

x2 = np.unique(np.asarray(probabilitySameClass))
y2 = [probabilitySameClass.count(element) for element in x2]

sumy = sum(y)
sumy2 = sum(y2)

y = [element/sumy for element in y]
y2 = [element/sumy2 for element in y2]

#Inter classe
plt.plot(x,y)
plt.fill(x,y)
plt.xlabel("Probability Error")
plt.ylabel("Probability/Density")
plt.title("Density Histogram as function of differents classes pair")
plt.show()

#Intra classe
plt.hist(pairProbabilityErrorSameClass(X,Y),bins=2,density=True)
plt.xlabel("Probability Error")
plt.ylabel("Probability/Density")
plt.title("Histogram of same classes pair")
plt.show()
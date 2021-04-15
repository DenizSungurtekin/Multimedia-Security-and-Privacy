import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage
from skimage.measure import block_reduce
import glob # Used to read all image in a directory
import numpy as gfg
import random
import copy
# 1 Encryption

#
# # Exercice 1
body = cv2.imread("liftingbody.png",0) #Read image without colors
height,width = body.shape
print("The image has height: ",height)
print("The image has width: ",width)


# Construction to an array containing all possible position index
permutationMatrix = np.asarray([[[i,j] for j in range(width)] for i in range(height)])
#We shuffle this array
np.random.shuffle(permutationMatrix)

#Then we just resize it into a matrix 512x512x2
permutationMatrix = permutationMatrix.reshape((height,width,2))


#### ---- SAVE AND LOAD THE PERMUTATION MATRIX UNCOMMENT TO USE --------- # The matrix will change at each run
# # We reshape it to 2d matrix to save it into txt format each pair of index are displayed individualy but next to each other
# np.savetxt('permutationMatrix.txt',permutationMatrix.reshape(permutationMatrix.shape[0],-1),fmt='%.0f')
#
# # We can load it as a 3d matrix again  with this part of code
# loaded_arr = gfg.loadtxt("permutationMatrix.txt")
# load_original_arr = loaded_arr.reshape(
#     loaded_arr.shape[0], loaded_arr.shape[1] // permutationMatrix.shape[2], permutationMatrix.shape[2])
### -------------------####

# We will now do the permutation
def permute(image,permutationMatrix):
    height,width = image.shape
    return np.asarray([image[permutationMatrix[i][j][0]][permutationMatrix[i][j][1]]for j in range(width) for i in range(height)])


permutedImage = np.asarray([body[permutationMatrix[i][j][0]][permutationMatrix[i][j][1]]for j in range(width) for i in range(height)])
# Reshape it in the image format
permutedImage = permutedImage.reshape((height,width))


#Now we plot the two images and the correspondig histogram

numpy_horizontal = np.hstack((body, permutedImage))
window2 = 'Original VS Permuted'
cv2.imshow(window2,numpy_horizontal) # Show image
cv2.waitKey(0)

histOriginal = cv2.calcHist([body],[0],None,[256],[0,256])
histPermuted = cv2.calcHist([permutedImage],[0],None,[256],[0,256])

fig,(ax1,ax2) = plt.subplots(1,2)
fig.suptitle("Original Histogram VS Permuted Histogramm ")
ax1.plot(histOriginal,color="blue")
ax2.plot(histPermuted,color="blue")
plt.show()



## Ex2

## functions to do a random but deterministic shuffle to be able to do the inverse permutation
def getperm(l):
    seed = len(l) #Random seed based on lenght of the image
    random.seed(seed)
    perm = list(range(len(l)))
    random.shuffle(perm)
    random.seed() # optional, in order to not impact other code based on random
    return perm

def shuffle(l):
    perm = getperm(l)
    l[:] = [l[j] for j in perm]

def unshuffle(l):
    perm = getperm(l)
    res = [None] * len(l)
    for i, j in enumerate(perm):
        res[j] = l[i]
    l[:] = res



#Function to shuffle and unshuffle image
def shuffleImage(image):
    N,M = image.shape
    vectorImage = np.reshape(image, (N * M)).tolist()
    shuffle(vectorImage)
    return np.reshape(np.asarray(vectorImage,dtype=np.uint8),(512,512))

def unshuffleImage(image):
    N,M = image.shape
    vectorImage = np.reshape(image, (N * M)).tolist()
    unshuffle(vectorImage)
    return np.reshape(np.asarray(vectorImage,dtype=np.uint8),(512,512))



body = cv2.imread("liftingbody.png",0) #Read image without colors
height,width = body.shape


def blockLoss(image):
    N,M = image.shape

    #Size of the block
    randN = random.randint(1,N)
    randM = random.randint(1, M)

    #TopLeftPosition of the block
    x = random.randint(1,N-randN)
    y = random.randint(1,N - randM)

    #Set to 0 the given area
    image[x:x+randN,y:y+randM] = 0

    return image

#Ex3
#Compute all images
shuffledImage = shuffleImage(body)
lossImage = blockLoss(shuffledImage)
unshuffledImage = unshuffleImage(lossImage)


numpy_horizontal = np.hstack((body, shuffledImage))
window2 = 'Original VS Permuted'
cv2.imshow(window2,numpy_horizontal) # Show image
cv2.waitKey(0)


numpy_horizontal = np.hstack((lossImage,unshuffledImage))
window2 = "BlockLoss VS BackPermutation"
cv2.imshow(window2,numpy_horizontal) # Show image
cv2.waitKey(0)

# Ex 4


body = cv2.imread("liftingbody.png",0) #Read image without colors

#Generate noise for an given image between an interval -> Uniform distribution

def generateNoisyImage(image,low,high):
    #Perfom conversion in double datatype before
    N,M = image.shape
    noisedMatrix = np.around(np.reshape(np.asarray(np.random.uniform(low,high,N*M)),(512,512)))
    return (image+noisedMatrix).astype(np.uint8)


def plotImageAndHistogram(image):
    lows = [-1,-5,-10,-15]
    high = [1,5,10,15]
    psnrs = []
    for i in range(len(lows)):
        noisedImage = generateNoisyImage(image,lows[i],high[i])
        Hist = cv2.calcHist([image],[0],None,[256],[0,256])
        fig = plt.figure()
        title = 'Interval: ',high[i]
        plt.title(title)
        plt.plot(Hist, color="blue")
        plt.show()
        window2 = "NoisedImage"
        cv2.imshow(window2, noisedImage)  # Show image
        cv2.waitKey(0)
        psnrs.append(cv2.PSNR(image,noisedImage))
    x = [1,5,10,15]
    plt.plot(x, psnrs)
    plt.xlabel("Interval value")
    plt.ylabel("PSNR")
    plt.title("PSNR evolution")
    plt.show()

plotImageAndHistogram(body)


# 2  Classical Cryptography##
# See jupyter notebook


### 3 Basic Data Hiding

# Show if your are uising little endian
from sys import byteorder
print(" ")
print("The system use ",byteorder," endian")

# Load image S and C and print shape ## Color order are BGR
cover = cv2.imread("lena.png")
secret = cv2.imread("baboon.png")

N,M,C = cover.shape
N2,M2,C2 = secret.shape

print(" ")
print("Cover dimensions: ",N," ",M," ",C)
print("Secret dimensions: ",N2," ",M2," ",C2)
print(" ")


## Function to modify a bits of a integer at a certain position

def modifyBit(integer,  position,  bitsValue): #position 0 is the smallest bit
    mask = 1 << position
    return (integer & ~mask) | ((bitsValue << position) & mask)


#Function to extract one bit from an integer at position p

def bitExtracted(integer, position):
    string = "{:08b}".format(integer)
    string = string[::-1] # to take the position th litle bits
    bit = string[position]
    return int(bit)

## Implementation of two functions insertion and recover


# Insertion of secret image
def insertion(cover,secret):
    N, M, C = cover.shape

    # Index of the
    blueIndex = [0, 1, 2, 3]
    # Index of which bits is taken in the secret minus 1
    blueIndexSecret = [4, 5, 6, 7]

    greenIndex = [0, 1]
    greenIndexSecret = [6, 7]

    redIndex = [0, 1, 2]
    redIndexSecret = [5, 6, 7]

    cover[:, :, 0] = np.asarray([[modifyBit(modifyBit(modifyBit(modifyBit(cover[i, j, 0], blueIndex[0], bitExtracted(secret[i, j, 0], blueIndexSecret[0])),blueIndex[1], bitExtracted(secret[i, j, 0], blueIndexSecret[1])), blueIndex[2],bitExtracted(secret[i, j, 0], blueIndexSecret[2])), blueIndex[3],bitExtracted(secret[i, j, 0], blueIndexSecret[3])) for j in range(M)] for i in range(N)])
    cover[:, :, 1] = np.asarray([[modifyBit(modifyBit(cover[i, j, 0], greenIndex[0], bitExtracted(secret[i, j, 0], greenIndexSecret[0])), greenIndex[1],bitExtracted(secret[i, j, 0], greenIndexSecret[1])) for j in range(M)] for i in range(N)])
    cover[:, :, 2] = np.asarray([[modifyBit(modifyBit(modifyBit(cover[i, j, 0], redIndex[0], bitExtracted(secret[i, j, 0], redIndexSecret[0])), redIndex[1],bitExtracted(secret[i, j, 0], redIndexSecret[1])), redIndex[2],bitExtracted(secret[i, j, 0], redIndexSecret[2])) for j in range(M)] for i in range(N)])

    return cover


#Function taking some bits and return the corresponding integer, this bits being the most significant bits in one byte
def mostBits(bits):
    empty =""
    for element in bits:
        empty += str(element)

    empty = empty[::-1]

    size = len(empty)
    while size !=8:
        empty +=str(0)
        size += 1
    return int(empty,2)



## Secret extraction
def extraction(stego):
    N, M, C = cover.shape

    # Index of the modified bits in cover
    blueIndex = [0, 1, 2, 3]
    greenIndex = [0, 1]
    redIndex = [0, 1, 2]

    stego[:, :, 0] = [[mostBits([bitExtracted(stego[i,j,0],p) for p in blueIndex]) for j in range(M)] for i in range(N)]
    stego[:, :, 1] = [[mostBits([bitExtracted(stego[i,j,1],p) for p in greenIndex]) for j in range(M)] for i in range(N)]
    stego[:, :, 2] = [[mostBits([bitExtracted(stego[i,j,2],p) for p in redIndex]) for j in range(M)] for i in range(N)]

    return stego

stego = insertion(cover,secret)
window2 = "Stego Image"
cv2.imshow(window2,stego) # Show image
cv2.waitKey(0)

recover = extraction(stego)
window3 = "Recovered Image"
cv2.imshow(window3,recover) # Show image
cv2.waitKey(0)
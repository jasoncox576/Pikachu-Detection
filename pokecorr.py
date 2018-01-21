from PIL import Image
import numpy as np
from correlation import cross_correlate
import math
from scipy import signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def plot_3d(Z, length, width):

    Y = np.arange(0, width, 1)
    X = np.arange(0, length, 1)

    fig = plt.figure()
    ax = Axes3D(fig)

    Y, X = np.meshgrid(Y, X)
    ax.plot_surface(Y, X, Z, rstride=1, cstride=1, cmap="hot")

    plt.show()


def plot_2d(corrs):

    plt.imshow(corrs)
    plt.show()


    
def calculate_dim(image):

    height = len(image)
    width = len(image[0])
    area = height * width
    return height, width, area



def mean_blur(image, kernel_len):
    
    """ In this mean blur, it is purposeless for an actual
    kernel to exist to do the convolution because it is simply
    an nd array filled with 1's. Thus you can achieve the same
    effect just by summing the pixel values and dividing by the 'kernel area'"""
    imageHeight, imageWidth, imageArea = calculate_dim(image)
    kernel_area = kernel_len**2

    for Y in range(imageHeight):
        for X in range(imageWidth):
            for Z in range(3):
                weighted_avg = 0  # Initialize sum variable that will be divided by kernel_len**2
                for y in range(kernel_len):
                    for x in range(kernel_len):
                        if ((Y-(kernel_len/2)+y) < imageHeight) and ((X-(kernel_len/2)+x) < imageWidth):
                            weighted_avg += image[int(Y-(kernel_len/2)+y)][int(X-(kernel_len/2)+x)][Z]
                image[Y][X][Z] = weighted_avg / kernel_area
            
    return image





def normalize(array):
    
    """Z-normalization used to bring the array to a range between -1 and 1.
    An issue was encountered in using homeMade_corr algorithm to calculate cross correlation
    in which the white blank space in the image would have the highest correlation because of
    the multiplication with extremely high max of (255, 255, 255). This method is an attempt
    to bring the values to a range that gives a punishment to total correlation if a positive
    value is multiplied by a negative value."""

    mean = np.mean(array)
    std = np.std(array)

    arrayHeight, arrayWidth, arrayArea = calculate_dim(array)
    for row in range(arrayHeight):
        for col in range(arrayWidth):
            for val in range(3):
                array[row][col][val] = (array[row][col][val] - mean)/std
    return array



def inverse_normalize(array):

    mean = np.mean(array)
    std = np.std(array)

    for row in range(len(array)):
        for col in range(len(array[row])):
            for val in range(3):
                array[row][col][val] = (array[row][col][val] * std) + mean
    return array








def library_corr(kernel, image):
    
    """ Note: The numpy correlate2d function is not guaranteed to work as
    the output size is different from the size of the inputs, unlike
    homeMade_corr function."""


    kernel = kernel.astype(float)
    image = image.astype(float)

    kernel = normalize(kernel)
    image = normalize(image)


    kernelHeight, kernelWidth, kernelArea = calculate_dim(kernel)
    imageHeight, imageWidth, imageArea = calculate_dim(image)


    kernel_red = np.zeros((kernelHeight, kernelWidth))
    kernel_blue = np.zeros((kernelHeight, kernelWidth))
    kernel_green = np.zeros((kernelHeight, kernelWidth))

    image_red = np.zeros((imageHeight, imageWidth))
    image_blue = np.zeros((imageHeight, imageWidth))
    image_green = np.zeros((imageHeight, imageWidth))


    for Y in range(len(kernel)):
        for X in range(len(kernel[Y])):
            kernel_red[Y][X] = kernel[Y][X][0]
            kernel_blue[Y][X] = kernel[Y][X][1]
            kernel_green[Y][X] = kernel[Y][X][2]

    for Y in range(len(image)):
        for X in range(len(image[Y])):
            image_red[Y][X] = image[Y][X][0]
            image_blue[Y][X] = image[Y][X][1]
            image_green[Y][X] = image[Y][X][2]

    red_corrs = signal.correlate2d(image_red, kernel_red, 'same')
    blue_corrs = signal.correlate2d(image_blue, kernel_blue, 'same')
    green_corrs = signal.correlate2d(image_green, kernel_green, 'same')

    # Does this hamper the accuracy?

    averaged_corrs = (red_corrs + blue_corrs + green_corrs) / 3



    maxcoords = (0,0)
    maxcorr = 0
    for x in range(len(averaged_corrs)):
        for y in range(len(averaged_corrs[x])):
            if averaged_corrs[x][y] > maxcorr:
                maxcorr = averaged_corrs[x][y]
                maxcoords = (x, y)

    print("Max correlation:", maxcorr)
    print("Max coordinates:", maxcoords)

    return maxcorr, maxcoords, averaged_corrs




def homeMade_corr(kernel, image):



    kernel = kernel.astype(float)
    image = image.astype(float)

    maxcorr = 0
    maxcoords = ()

    kernel = normalize(kernel)
    image = normalize(image)

    imageHeight, imageWidth, imageArea = calculate_dim(image)
    corrs = np.zeros((imageHeight, imageWidth))
    opaque_pixels = 0
    for y in range(len(kernel)):
        for x in range(len(kernel[y])):
            if float(kernel[y][x][3]) == 255.0:
                opaque_pixels += 1

    for Y in range(len(image) - len(kernel)):
        for X in range(len(image[Y]) - len(kernel[0])):
            corr = 0
            for y in range(len(kernel)):
                for x in range(len(kernel[y])):
                    if float(kernel[y][x][3]) == 255.0:
                        for i in range(3):
                            corr += (kernel[y][x][i] * image[y+Y][x+X][i])
            corr = corr / 3 / opaque_pixels
            corrs[Y][X] = corr
            if corr > maxcorr:
                maxcorr = corr
                maxcoords = (Y, X)
                print(maxcorr, "at", maxcoords)

    print("Point of maxcorr is:", maxcoords)

    
    return maxcorr, maxcoords, corrs



def multimax_search(corrs):


    multimax_coords = []
    argmax = np.unravel_index(np.argmax(corrs), corrs.shape)
    maxcorr = corrs[argmax[0], argmax[1]]
    threshold = .99*maxcorr
    
    for Y in range(len(corrs)):
        for X in range(len(corrs[Y])):
            surrounds = []
            if (Y==0 and X==0): # top left corner
                surrounds.append(corrs[Y+1, X])
                surrounds.append(corrs[Y, X+1])
            elif (Y==len(corrs)-1 and X==0): # bottom left corner
                surrounds.append(corrs[Y-1, X])
                surrounds.append(corrs[Y, X+1])
            elif (Y==0 and X==len(corrs[Y])-1): # top right corner
                surrounds.append(corrs[Y, X-1])
                surrounds.append(corrs[Y+1, X])
            elif (Y==len(corrs)-1 and X==len(corrs[Y])-1): # bottom right corner
                surrounds.append(corrs[Y-1, X])
                surrounds.append(corrs[Y, X-1])
            elif (X==0 or X==len(corrs[Y])-1) and not(Y==0 or Y==len(corrs)-1): # Top or bottom row
                surrounds.append(corrs[Y+1, X])
                surrounds.append(corrs[Y-1, X])
                if X==0: surrounds.append(corrs[Y, X+1])
                else: surrounds.append(corrs[Y, X-1])
            elif (Y==0 or Y==len(corrs)-1) and not(X==0 or X==len(corrs[Y])-1): # Left or right-most column
                surrounds.append(corrs[Y, X+1])
                surrounds.append(corrs[Y, X-1])
                if Y==0: surrounds.append(corrs[Y+1, X])
                else: surrounds.append(corrs[Y-1, X])
            else: # Any element in between
                surrounds.append(corrs[Y, X+1])
                surrounds.append(corrs[Y, X-1])
                surrounds.append(corrs[Y+1, X])
                surrounds.append(corrs[Y-1, X])
            for x in surrounds:
                if x > corrs[Y][X]:
                    break
            if corrs[Y][X] >= threshold:
                multimax_coords.append((Y, X))

    return multimax_coords    


def drawBoundingBox(kernel, image, maxcoords):
    
    kernelHeight, kernelWidth, kernelArea = calculate_dim(kernel)
    
    
    for current_max in range(len(maxcoords)):
        try:
            for x in range(kernelHeight):
                if((x==0) or (x==kernelHeight-1)):
                    for y in range(len(kernel[x])):
                        image[maxcoords[current_max][0]+x][maxcoords[current_max][1]+y] = [0,0,0,255]
                else:
                    image[maxcoords[current_max][0]+x][maxcoords[current_max][1]] = [0,0,0,255]
                    image[maxcoords[current_max][0]+x][maxcoords[current_max][1]+(kernelWidth-1)] = [0,0,0,255]

        except IndexError:
            print("The bounding box is out of bounds. This likely means that the template represented by the max does not exist entirely within the image \n or there is something wrong with the correlation algorithm being used.")
    return image




def output_image(image):

    imageHeight, imageWidth, imageArea = calculate_dim(image)

    output_img = Image.new('RGBA', (imageHeight, imageWidth))

    output_data = []

    for x in range(len(image)): 
        for y in range(len(image[x])):
            output_data.append(tuple(image[x][y]))    

    output_data = tuple(output_data)
    output_img.putdata(output_data)
    output_img.save('out.bmp')










def main():

    pikachu = Image.open('pikachu.bmp')
    image = Image.open('image.bmp')


    kernel = np.array(list(pikachu.getdata()))
    image = np.array(list(image.getdata()))

    kernel = np.reshape(kernel, (-1, math.sqrt(len(kernel)), 4))
    image = np.reshape(image, (-1, math.sqrt(len(image)), 4))
    
        
    image = mean_blur(image, 3)

    maxcorr, maxcoords, corrs = homeMade_corr(kernel, image)
    multimaxes = multimax_search(corrs)
    for x in multimaxes:
        print(x)
    image = drawBoundingBox(kernel, image, multimaxes)
    output_image(image)
    
    
    plot_3d(corrs, len(image), len(image[0]))
 




main()












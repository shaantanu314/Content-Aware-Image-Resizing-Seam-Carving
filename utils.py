import copy
import numpy as np

def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
        
        
def convolve2D(image, kernel,wpadding=0,hpadding=0, strides=1):
    kernel = np.flipud(np.fliplr(kernel))

    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    xOutput = int(((xImgShape - xKernShape + 2 * hpadding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * wpadding) / strides) + 1)

    if len(image.shape) == 2:
        output = np.zeros((xOutput, yOutput))
        imagePadded = np.zeros((image.shape[0] + hpadding*2, image.shape[1] + wpadding*2))
    else:
        output = np.zeros((xOutput, yOutput,image.shape[2]))
        imagePadded = np.zeros((image.shape[0] + hpadding*2, image.shape[1] + wpadding*2,image.shape[2]))

    imagePadded[int(hpadding):int(image.shape[0] + hpadding*1), int(wpadding):int(image.shape[1] + wpadding*1)] = image 

    for y in range(imagePadded.shape[1]):
        if y > imagePadded.shape[1] - yKernShape:
            break
        if y % strides == 0:
            for x in range(imagePadded.shape[0]):
                if x > imagePadded.shape[0] - xKernShape:
                    break
                try:
                    if x % strides == 0:
                      if len(image.shape) == 2:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                      else:
                        for i in range(image.shape[2]):
                          output[x,y,i] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape,i]).sum()
                except:
                    break

    return output
### inbuilt ###
import sys
#import sleep

### user defined ###
from SeamCarving import SeamCarving


if __name__ == '__main__':
    
    args = sys.argv[1:]
    
    protect_mask = False
    remove_mask = False
    
    filename = None
    
    energy = None
    
    removal = False
    insertion = False
    
    for i in range(len(args)):
        if args[i] == "-p":
            protect_mask = True
        if args[i] == "-f":
            filename = args[i+1]
        if args[i] == "-r":
            remove_mask = True
        if args[i] == "-e":
            energy = args[i+1]
        if args[i] == "-R":
            removal = True
        if args[i] == "-I":
            insertion = True  
    
    s = SeamCarving(filename,protect_mask,remove_mask,energy_func=energy)
    

    print(f"the size of image is {s.image.shape}")
    
    if removal == True:
        out_height = int(input("output height :"))
        out_width = int(input("output width :"))
        s.seam_removal(out_height,out_width)
    elif insertion == True:
        out_width = int(input("output width :"))
        s.seam_insertion(out_width)
    
    s.display_image()
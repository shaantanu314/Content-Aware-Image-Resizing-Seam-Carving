from utils import convolve2D,printProgressBar
import energy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.path as mplPath
import matplotlib.cm as cm
import numpy as np
import copy
import cv2
import sys
import time
from numba import jit
import warnings
warnings.filterwarnings('ignore')

class SeamCarving():
    
    def __init__(self,filename,protect_mask=False,remove_mask=False,energy_func="L1"):
        self.filename = filename
        self.original_image = self.load_image(self.filename)
        self.image = self.load_image(self.filename)
        self.protect_mask = protect_mask
        self.remove_mask = remove_mask
        self.ProtectMaskArray = []
        self.RemoveMaskArray =[]
        self.ImageMask = np.zeros((self.image.shape[:2]),dtype="int64")
        if energy_func == "L1":
            self.energy_func = energy.L1
        elif energy_func == "L2":
            self.energy_func = energy.L2
        elif energy_func == "Entropy":
            self.energy_func = energy.Entropy
        elif energy_func == "HoG":
            self.energy_func = energy.HoG
        elif energy_func == "forward_energy":
            self.energy_func = energy.forward_energy
        else:
            print("Invalid energy function.Will use forward_energy.")
            self.energy_func = energy.forward_energy
            
        if self.protect_mask:
            self.get_mask(self.image)
        
        if self.remove_mask:
            self.get_mask(self.image)
        
    def reset_state(self):
        self.image = self.original_image
    
    def load_image(self,filename):
        return cv2.imread(filename)
    
    def display_image(self):
        plt.figure(figsize=(15,20))
        plt.subplot(1,2,1)
        plt.imshow(self.original_image[:,:,::-1])
        plt.title('Original Image')
        plt.subplot(1,2,2)
        plt.imshow(self.image[:,:,::-1])
        plt.title('Resized Image')
        plt.show()
    
    

    def get_mask(self,image):
        img = copy.deepcopy(image)
        
        def handle_close(event):
            self.ProtectMaskArray = np.array(self.ProtectMaskArray)
            self.RemoveMaskArray = np.array(self.RemoveMaskArray)
            if self.protect_mask:
                poly_path = mplPath.Path(self.ProtectMaskArray)
                for row in range(self.image.shape[0]):
                    for col in range(self.image.shape[1]):
                        if(poly_path.contains_point(np.array([col,row]))):
                            self.ImageMask[row][col] = 1

            if self.remove_mask:
                poly_path = mplPath.Path(self.RemoveMaskArray)
                for row in range(self.image.shape[0]):
                    for col in range(self.image.shape[1]):
                        if(poly_path.contains_point(np.array([col,row]))):
                            self.ImageMask[row][col] = -1
            
            plt.imshow(self.ImageMask)
            plt.close()
            self.show_mask()
            
            
                
        
        def onclick(event):
            ix, iy = int(event.xdata), int(event.ydata)
            if self.protect_mask:
                self.ProtectMaskArray.append([ix, iy])
            if self.remove_mask:
                self.RemoveMaskArray.append([ix, iy])
        
        fig = plt.figure(figsize=(10,15))
        imgplot = plt.imshow(img[:,:,::-1])
        fig.canvas.mpl_connect('button_press_event',onclick)
        fig.canvas.mpl_connect('close_event',handle_close)
        plt.show()

    
    def show_mask(self):
        rem_points = None
        color = []
        if self.protect_mask:
            rem_points = self.ProtectMaskArray
            color = [0,255,0]
        if self.remove_mask:
            rem_points = self.RemoveMaskArray
            color = [0,0,255]
        img = copy.deepcopy(self.image)
        poly_path = mplPath.Path(rem_points)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if(poly_path.contains_point(np.array([j,i]))):
                    img[i][j] = color

        imgplot = plt.imshow(img[:,:,::-1])
        plt.scatter(rem_points[:,0],rem_points[:,1],c="r",s=10)
        plt.show()
 
  
    @jit
    def find_seam(self,mat,input_image):
        height , width ,_ = input_image.shape
        dp = copy.deepcopy(mat)
        for i in range(1,height):
            for j in range(width):
                min_path = dp[i-1][j]
                if(j!=0):
                    min_path = min(min_path,dp[i-1][j-1])
                if(j!=width-1):
                    min_path = min(min_path,dp[i-1][j+1])
                dp[i][j] += min_path

        seam_indices = np.empty(height,dtype=int)
        p_ind = np.argmin(dp[-1])
        seam_indices[-1] = p_ind
        for i in reversed(range(height-1)):
            c_val,c_ind = dp[i][p_ind],p_ind
            if(p_ind>0 and dp[i][p_ind-1]<c_val):
                c_val,c_ind = dp[i][p_ind-1],p_ind-1
            if(p_ind<width-1 and dp[i][p_ind+1]<c_val):
                c_val,c_ind = dp[i][p_ind+1],p_ind+1
            p_ind = c_ind
            seam_indices[i] = p_ind

        return seam_indices
    
    def update_mask(self,seam_indices,inMask,mode="Removal"):
        if mode == "Removal":
            outMask = np.zeros((inMask.shape[0],inMask.shape[1]-1),dtype="int64")
            for i,seam in enumerate(seam_indices):
                outMask[i] =  np.concatenate((inMask[i][:seam],inMask[i][seam+1:]))
            return outMask

    @jit  
    def remove_seam(self,seam_indices,input_image):
        image = copy.deepcopy(input_image)
        height , width ,_ = image.shape
        resized_image = np.empty((height,width-1,3),dtype=np.uint8)
        for i in range(height):
            resized_image[i] = np.concatenate((image[i][:seam_indices[i]],image[i][seam_indices[i]+1:]))
        
        image = resized_image
        return image 
     
    def seam_removal(self,target_height,target_width):
        prev_mode = "HEIGHT"
        curr_mode = "HEIGHT"
        total_moves = (self.image.shape[0]-target_height + self.image.shape[1]-target_width)
        iteration_no = 0
        while target_height < self.image.shape[0] or target_width < self.image.shape[1]:
            if target_width == self.image.shape[1] and curr_mode == "HEIGHT":
                curr_mode = "WIDTH"
            elif target_height == self.image.shape[0] and curr_mode == "WIDTH":
                curr_mode = "HEIGHT"
            if curr_mode != prev_mode:
                target_height,target_width = target_width , target_height
                if curr_mode == "WIDTH":
                    self.image = np.rot90(self.image,k=1,axes=(0,1))
                    self.ImageMask = np.rot90(self.ImageMask,k=1,axes=(0,1))
                else :
                    self.image = np.rot90(self.image,k=3,axes=(0,1))
                    self.ImageMask = np.rot90(self.ImageMask,k=3,axes=(0,1))
            energy_mat = self.energy_func(self.image) 
            energy_mat = energy_mat + self.ImageMask*100000
            
            seam_indices = self.find_seam(energy_mat,self.image) 
            test_img = copy.deepcopy(self.image)
            for cnt,seam in enumerate(seam_indices):
                test_img[cnt][seam] = [0,255,0]
            cv2.imwrite('Results/'+str(iteration_no)+'.jpg',test_img)
            self.image = self.remove_seam(seam_indices,self.image)
            self.ImageMask = self.update_mask(seam_indices,self.ImageMask,mode="Removal")
            prev_mode , curr_mode = curr_mode , prev_mode
            iteration_no = iteration_no +1
            
            printProgressBar(iteration_no, total_moves, prefix = 'Progress:', suffix = 'Complete', length = 50)
            

        if prev_mode == "WIDTH":
            self.image = np.rot90(self.image,k=3,axes=(0,1))
            self.ImageMask = np.rot90(self.ImageMask,k=3,axes=(0,1))

        return
        
    @jit
    def insert_seam(self,seam):
        height, width = self.image.shape[: 2]
        output = np.zeros((height, width + 1, 3),dtype="int64")
        for row in range(height):
            col = seam[row]
            for ch in range(3):
                if col == 0:
                    p = np.average(self.image[row, col: col + 2, ch])
                    p = int(p)
                    output[row, col, ch] = self.image[row, col, ch]
                    output[row, col + 1, ch] = p
                    output[row, col + 1:, ch] = self.image[row, col:, ch]
                else:
                    p = np.average(self.image[row, col - 1: col + 1, ch])
                    p = int(p)
                    output[row, : col, ch] = self.image[row, : col, ch]
                    output[row, col, ch] = p
                    output[row, col + 1:, ch] = self.image[row, col:, ch]
                    
        self.image = np.array(output)
        return 
    @jit
    def update_seams(self,remaining_seams,current_seam):
        output = []
        for seam in remaining_seams:
            seam[np.where(seam >= current_seam)] += 2
            output.append(seam)
        return output
    @jit
    def seam_insertion(self,target_width):
        '''
        Seam Insertion only supports width enlargement as of now. The Approach used to add seam doesn't 
        work well with the addition of both vertical and horizontal seams. Thus we have implemented the 
        function to insert vertical seams only.
        '''
        if target_width <self.image.shape[1]:
            print("Error: Target Width lesser than original image width")
            return
        temp_image = copy.deepcopy(self.image)
        temp_mask = copy.deepcopy(self.ImageMask)
        seam_record = []
        temp_height,temp_width,_ = temp_image.shape
        target_width = -target_width + 2*temp_width
        total_moves = temp_image.shape[1]-target_width
        iteration_no = 0
        while target_width < temp_image.shape[1]:
            energy_mat = self.energy_func(temp_image) + temp_mask * 100000
            seam_indices = self.find_seam(energy_mat,temp_image)
            seam_record.append(seam_indices)
            temp_image = self.remove_seam(seam_indices,temp_image)
            temp_mask = self.update_mask(seam_indices,temp_mask,mode="Removal")
            iteration_no = iteration_no + 1
            printProgressBar(iteration_no, total_moves, prefix = 'Progress:', suffix = 'Complete', length = 50)
                
        n = len(seam_record)
        for i in range(n):
            seam = seam_record.pop(0)
            self.insert_seam(seam)
            seam_record = self.update_seams(seam_record,seam)
            printProgressBar(i, n, prefix = 'Progress:', suffix = 'Complete', length = 50)
            

        return 
            
    
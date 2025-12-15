import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time

def lowlight(image_path):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	scale_factor = 12
	data_lowlight = Image.open(image_path)

 

	data_lowlight = (np.asarray(data_lowlight)/255.0)


	data_lowlight = torch.from_numpy(data_lowlight).float()

	h=(data_lowlight.shape[0]//scale_factor)*scale_factor
	w=(data_lowlight.shape[1]//scale_factor)*scale_factor
	data_lowlight = data_lowlight[0:h,0:w,:]
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)

	DCE_net = model.enhance_net_nopool(scale_factor).cuda()
	DCE_net.load_state_dict(torch.load('snapshots_Zero_DCE++/Epoch99.pth'))
	start = time.time()
	enhanced_image,params_maps = DCE_net(data_lowlight)

	end_time = (time.time() - start)

	print(end_time)
	image_path = image_path.replace('test_data','result_Zero_DCE++')

	result_path = image_path
	if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
		os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))
	# import pdb;pdb.set_trace()
	torchvision.utils.save_image(enhanced_image, result_path)
	return end_time




if __name__ == '__main__':
    with torch.no_grad():

        # 你的图片目录
        in_dir = "/home/pearson/Projects/ECE253/dataset"

        # 输出目录
        out_dir = "/home/pearson/Projects/ECE253/results/ZeroDCE"
        os.makedirs(out_dir, exist_ok=True)

        # 支持 jpg / jpeg / png 等格式
        exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]

        # 收集所有待测图片
        test_list = []
        for ext in exts:
            test_list.extend(glob.glob(os.path.join(in_dir, ext)))

        sum_time = 0

        for image in test_list:
            print("Processing:", image)
            sum_time += lowlight(image)

        print("Total time:", sum_time)
        
        
# if __name__ == '__main__':

# 	with torch.no_grad():

# 		filePath = 'data/test_data/'	
# 		file_list = os.listdir(filePath)
# 		sum_time = 0
# 		for file_name in file_list:
# 			test_list = glob.glob(filePath+file_name+"/*") 
# 		    for image in test_list:

# 				print(image)
# 				sum_time = sum_time + lowlight(image)

# 		print(sum_time)
		


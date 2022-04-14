# slective search生成训练集，IoU>=0.5为positive
import cv2
import pandas as pd
from utils import cal_IoU
import os
import numpy as np
import multiprocessing
import download


class Selective_search():
    def __init__(self, dir_path, ):
        '''
        Args:
            dir_path: 训练或验证数据所在文件夹
        '''
        self.csv_path = os.path.join(dir_path, 'label.csv')
        self.imgs_path = os.path.join(dir_path, 'images')
        self.flag = dir_path.split('_')[-1]
        self.num_per_image = 8
        
    @staticmethod
    def cal_pro_region(img_path):
        '''计算每张图片的proposal region
        Args:
            img_path: 图片所在路径
        Returns:
            np.array: proposal region的坐标, 大小为num*4, 4列分别[xmin, ymin, xmax, ymax]
        '''
        try:
            ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        except AttributeError:
            raise Exception('需要安装opencv-contrib-python, 安装前请先删除原有的opencv-python')
        ss.setBaseImage(cv2.imread(img_path))
        ss.switchToSelectiveSearchFast()
        rects = ss.process()
        rects[:, 2] += rects[:, 0]
        rects[:, 3] += rects[:, 1] 
        return rects

    def save(self, num_workers=1):
        '''可选多进程计算proposal regions
        Args:
            num_workers: 进程个数
        '''
        self.csv = pd.read_csv(self.csv_path, header=0, index_col=None)
        self.positive_path = './data/ss/' + self.flag + '/banana/'
        self.negative_path = './data/ss/' + self.flag + '/background/'
        os.makedirs(self.positive_path, exist_ok=True)
        os.makedirs(self.negative_path, exist_ok=True)
        index = self.csv.index.to_list()
        span = len(index)//num_workers
        print(f'=======开始计算proposal regions of {self.flag} imgs...=======')
        for i in range(num_workers):
            if i != num_workers-1:
                multiprocessing.Process(target=self.save_pr, 
                            kwargs={'index': index[i*span:(i+1)*span]}).start()
            else:
                multiprocessing.Process(target=self.save_pr, 
                            kwargs={'index': index[i*span:]}).start()
              
    
    def save_pr(self, index):
        '''根据索引保存该图片proposal regions坐标-xmin, ymin, xmax, ymax
        Args:
            index(list): 索引
        '''
        for row in index:
            img_name = self.csv.iloc[row, 0]
            gt_box = self.csv.iloc[row, 2:].values
            img_path = os.path.join(self.imgs_path, img_name)
            region_pro = self.cal_pro_region(img_path)    # proposal region坐标--num*4大小的np.array
            IoU = cal_IoU(region_pro, gt_box)

            locs_p = region_pro[np.where(IoU>=0.5)]  # IoU超过0.5，positive
            locs_n = region_pro[np.where((IoU<0.5) & (0.1<IoU))] # IoU<0.5，negative
            
            img = cv2.imread(img_path)
            for (j, loc) in enumerate(locs_p):
                crop = img[loc[1]:loc[3], loc[0]:loc[2], :]
                crop_img = self.positive_path + img_name.split('.')[0]+'_'+str(j)+'.png'
                with open('./data/ss_'+self.flag+'_loc.csv', 'a') as f:
                    f.writelines([crop_img, ',', str(loc[0]), ',', str(loc[1]), ',', str(loc[2]), ',', str(loc[3]), '\n'])
                cv2.imwrite(crop_img, crop)
                if j==self.num_per_image-1:
                    break
            print(f'{img_name}: {j+1}个positive', end='\t')

            for (j, loc) in enumerate(locs_n):
                crop = img[loc[1]:loc[3], loc[0]:loc[2], :]
                crop_img = self.negative_path + img_name.split('.')[0]+'_'+str(j)+'.png'
                cv2.imwrite(crop_img, crop)
                if j==self.num_per_image-1:
                    break
            print(f'{j+1}个negative')


if __name__ == '__main__':
    download.download_extract()
    Selective_search('./data/banana-detection/bananas_val').save(4)
    Selective_search('./data/banana-detection/bananas_train').save(4)
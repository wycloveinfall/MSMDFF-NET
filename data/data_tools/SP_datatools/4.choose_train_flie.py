# -*-coding:utf-8-*-

import os
from tqdm import trange
import shutil
if __name__ == '__main__':

    print('cwd',os.getcwd())
    root_dir = os.getcwd()+"/data/train/SpaceNet"
    for _file in os.listdir(root_dir):
        city_file = root_dir.replace("\\",'/') + '/' + _file
        RGB_8bit = city_file + '/' + 'RGB-8bit'
        center_line_file = city_file + '/' + 'label_center_line'
        out_image_file = city_file + '/' + 'train_src'
        os.makedirs(out_image_file,exist_ok=True)
        with trange(len(os.listdir(center_line_file))) as t:
            for __file,index in zip(os.listdir(center_line_file),t):
                __file = __file.replace('_roads_label_','_PS-RGB_').replace('.tif','_sat.tif')
                shutil.move(RGB_8bit + '/' + __file,out_image_file + '/' + __file)
        shutil.rmtree(RGB_8bit,ignore_errors=True)
    print('finished!!')

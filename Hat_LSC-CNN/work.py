import os
img_root = r'/home/ubuntu/dataset/Train'
img_list = sorted([name for name in os.listdir(img_root)])
for name in img_list:
    img_path = os.path.join(img_root,name)
    new_path = os.path.join(img_root,'qnrf_train_'+name.split('_')[1].replace('jpg','jpeg'))
    os.rename(img_path,new_path)
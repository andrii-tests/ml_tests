import os
import cv2
import shutil
from os.path import join
from random import choices
import xml.etree.ElementTree as ET
from shutil import copyfile
import xml.dom


vis = False
copy = True

base_dir = '/home/lv-user187/Desktop/heads/datasets/HollywoodHeads/'
imgs_dir = join(base_dir, 'JPEGImages')
lbls_dir = join(base_dir, 'Annotations')
new_dir = join(base_dir, 'Part_Heads_5')

lbls = os.listdir(join(base_dir, lbls_dir))
selected = choices(range(224740), k=int(len(lbls)/5))

for i in selected:
    lb_path_name = join(lbls_dir, lbls[i])

    tree = ET.parse(lb_path_name)
    root = tree.getroot()
    filename = root.find('filename').text

    if copy:
        copyfile(join(imgs_dir, filename), join(new_dir, filename))
        copyfile(lb_path_name, join(new_dir, lbls[i]))

    if vis:
        img = cv2.imread(join(imgs_dir, filename))
        for boxes in root.iter('object'):
            xmin, ymin, xmax, ymax = [int(i.text.replace('.', ',')) for i in boxes.find('bndbox')]

            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0))
        cv2.imshow(lbls[i], img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

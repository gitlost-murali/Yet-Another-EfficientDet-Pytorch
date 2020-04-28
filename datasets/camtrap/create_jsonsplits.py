import json
from tqdm import tqdm # For checking progress
import os, sys # For arguments
from glob import glob # For listing filenames
from cv2 import imread # For finding dimensions

split_type = sys.argv[1] # train / val / test

if not os.path.exists('annotations/'): os.mkdir('annotations/')
if os.path.exists(f'annotations/instances_{split_type}.json'): sys.exit("File already exists...\nQuitting the program....")

split_filenames = [ os.path.basename(imgname) for imgname in glob(f"{split_type}/*")]

id2categories = {1:'animal',2:'person',3:'group',4:'vehicle'}
categories2id = dict((v,k) for (k,v) in id2categories.items())
'''
Go through each instance in bboxes_inc_empty_20200325.json.
Keep a counter for index count.
filename = download_id + .png
'''

coco_camtrap = {"info": {"description": "", "url": "", "version": "", "year": 2020, "contributor": "", "date_created": "2020-04-14 01:45:18.567988"}, "licenses": [{"id": 1, "name": 'null', "url": 'null'}], "categories": [{"id": 1, "name": f"{id2categories[1]}", "supercategory": "None"}, {"id": 2, "name": f"{id2categories[2]}", "supercategory": "None"}, {"id": 3, "name": f"{id2categories[3]}", "supercategory": "None"}, {"id": 4, "name": f"{id2categories[4]}", "supercategory": "None"}]}

images = []
imageid = 1
annotations = []
annotationid = 1

'''
image = {"id":1,"file_name":f".","height":250,"width":500}
images.append(image)

annotation = {"id":1,"image_id":94,"category_id":2,"bbox":[0.695,0.227,0.288,0.455],"iscrowd":0}
annotations.append(annotation)
annotation = {"id":2,"image_id":94,"category_id":1,"bbox":[0.385,0.27,0.288,0.455],"iscrowd":0}
annotations.append(annotation)

coco_camtrap['images'] = images
coco_camtrap['annotations'] = annotations
'''

with open("bboxes_inc_empty_20200325.json",'r') as fh:
    instances = json.load(fh)

for instance in tqdm(instances,desc='Progress:'):
    filename = instance['download_id'] + '.jpg'
    if not filename in split_filenames: continue # Skip if the image is from another split type
    try:
        height, width, _ = imread(f'{split_type}/{filename}').shape
    except Exception as e:
        print(filename)
        sys.exit("Image reading error.. Exiting the program",e)

    image = {"id":imageid,"file_name":f"{filename}","height":height,"width":width}
    labels = instance['bbox']
    for label in labels:
        category = label['category']    
        category_id = categories2id[category]
        bbox = [label['bbox'][0]*width,label['bbox'][1]*height,label['bbox'][2]*width,label['bbox'][3]*height]
        annotation = {"id":annotationid,"image_id":imageid,"category_id":category_id,"bbox":bbox,"iscrowd":0}
        annotations.append(annotation)
        annotationid+=1

    images.append(image)
    imageid+=1

coco_camtrap['images'] = images
coco_camtrap['annotations'] = annotations

with open(f'annotations/instances_{split_type}.json',"w") as fh:
    json.dump(coco_camtrap,fh)
import json
from tqdm import tqdm

id2categories = {1:'animal',2:'person',3:'group',4:'vehicle'}
categories2id = dict((v,k) for (k,v) in id2categories.items())
'''
Go through each instance in bboxes_inc_empty_20200325.json.
Keep a counter for index count.
filename = download_id + .png
for i in instance:
    categoryname = instance[i]['bbox']['category']
    coco_camtrap['categories']
'''
coco_camtrap = {"info": {"description": "", "url": "", "version": "", "year": 2020, "contributor": "", "date_created": "2020-04-14 01:45:18.567988"}, "licenses": [{"id": 1, "name": 'null', "url": 'null'}], "categories": [{"id": 1, "name": f"{id2categories[1]}", "supercategory": "None"}, {"id": 2, "name": f"{id2categories[2]}", "supercategory": "None"}]}

images = []
imageid = 1
annotations = []
annotationid = 1

'''
image = {"id":1,"file_name":f"."}
images.append(image)

annotation = {"id":1,"image_id":94,"category_id":2,"bbox":[0.695,0.227,0.288,0.455],"iscrowd":0}
annotations.append(annotation)
annotation = {"id":2,"image_id":94,"category_id":1,"bbox":[0.385,0.27,0.288,0.455],"iscrowd":0}
annotations.append(annotation)

coco_camtrap['images'] = images
coco_camtrap['annotations'] = annotations
print(coco_camtrap)
'''

with open("bboxes_inc_empty_20200325.json",'r') as fh:
    instances = json.load(fh)

for instance in tqdm(instances,desc='Progress:'):
    filename = instance['download_id'] + '.jpg'
    image = {"id":imageid,"file_name":f"{filename}"}
    labels = instance['bbox']
    for label in labels:
        category = label['category']    
        category_id = categories2id[category]
        bbox = label['bbox']
        annotation = {"id":annotationid,"image_id":imageid,"category_id":category_id,"bbox":bbox,"iscrowd":0}
        annotations.append(annotation)
        annotationid+=1

    images.append(image)
    imageid+=1

coco_camtrap['images'] = images
coco_camtrap['annotations'] = annotations

with open("cocoCamtrap_mdv4.json","w") as fh:
    json.dump(coco_camtrap,fh)
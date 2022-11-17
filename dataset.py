import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt


# get group ids
def get_groups(image):
    id = image['id']
    group_set = set()
    try:
        for point in image.find_all('points'):
            group_set.add(point['group_id'])
        for box in image.find_all('box'):
            group_set.add(box['group_id'])
    except Exception as e:
        print( f'image {id} has no group ids' )
        return None

    return group_set

def create_pepper(group_id, image):

    pepper_bbox = None

    boxes = {
        'stem' : None,
        'body' : None,
    }

    kps = {
        'right_shoulder' : [0,0,0],
        'left_shoulder' : [0,0,0],
        'center_shoulder' : [0,0,0],
        'stem' : [0,0,0],
        'body' : [0,0,0],
    }

    # visibility:
    #   0: not labeled
    #   1: labeled, but not visible
    #   2: labeled, visible

    for point in image.find_all('points'):
        if point['group_id'] != group_id:
            continue
        label = point['label']
        points = point['points'] # should only be 1 in here
        xy_str = points.split(',')
        x = float(xy_str[0])
        y = float(xy_str[1])
        kps[label] = [x, y, 2]

    for box in image.find_all('box'):
        if box['group_id'] != group_id:
            continue
        label = box['label']
        boxes[label] = [
            float(box['xtl']),
            float(box['ytl']),
            float(box['xbr']),
            float(box['ybr']),
        ]    


    if boxes['stem'] is not None and boxes['pepper'] is not None:
        pepper_bbox = [
            min(boxes['stem'][0],boxes['pepper'][0]),
            min(boxes['stem'][1],boxes['pepper'][1]),
            max(boxes['stem'][2],boxes['pepper'][2]),
            max(boxes['stem'][3],boxes['pepper'][3]),
        ]

    elif boxes['pepper'] is not None:
        pepper_bbox = boxes['pepper']
    elif boxes['stem'] is not None:
        pepper_bbox = boxes['stem']
    

    area = (pepper_bbox[2] - pepper_bbox[0]) * (pepper_bbox[3] - pepper_bbox[1])

    
    kp_array = []
    kp_array.append(kps['right_shoulder'])
    kp_array.append(kps['left_shoulder'])
    kp_array.append(kps['center_shoulder'])
    kp_array.append(kps['stem'])
    kp_array.append(kps['body'])

    annotation = {
        "id" : group_id, # int,
        "image_id" : image['id'],# int,
        "category_id" : 1, # int,
        "segmentation" : None, # RLE or [polygon],
        "area" : area, # float,
        "bbox" : pepper_bbox, # [x,y,width,height],
        "iscrowd" : 0, #0 or 1,
        "keypoints": kp_array,# [x1,y1,v1,...],
        "num_keypoints" : 5, # : int,
    }

    return annotation


# pass beaustiful soup image object
def parse_image(image):

    image_data = {
        "id" : image['id'], # int,
        "width" : image['width'], # int,
        "height" : image['height'], # int,
        "file_name" : image['name'], #str,
        "license" : 0, # int,
        "flickr_url" : "", # str,
        "coco_url" : "", #str,
        "date_captured": None, #datetime,
    }

    groups = get_groups(image)
    if groups is None:
        return image_data, None

    annotations = []
    for group_id in groups:
        pepper = create_pepper(group_id, image)
        annotations.append(pepper)



    return image_data, annotations


# parse the annotations file and return a organized object
def parse_annotations(file_path):

    images = []
    annotations = []

    with open(file_path, 'r') as xml_doc:
        soup = BeautifulSoup(xml_doc, 'lxml')
        #print(soup.prettify())

        for image in soup.find_all('image'):
            im, anno = parse_image(image)
            images.append(im)
            annotations.append(anno)

    # print(len(images))
    # print(len(annotations))

    # print(images[0])
    # print(annotations[0])

    return images, annotations

class PepperDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.images, self.annotations = parse_annotations(annotations_file)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        # get the image
        img_path = os.path.join(
            self.img_dir, 
            (self.images[idx])['file_name'])
        image = read_image(img_path)

        # get the annotations
        annos = self.annotations[idx]

        bboxes = []
        kps = []

        for anno in annos:
            bboxes.append(anno['bbox'])
            kps.append(anno['keypoints'])

        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return image, annos


if __name__ == "__main__":
    ds = PepperDataset('annotations.xml', 'images')

    im, anno = ds[0]

    plt.imshow(  im.permute(1, 2, 0)  )

    # wait for key press
    input()
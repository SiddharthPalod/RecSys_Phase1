import json
import numpy as np
import os
import argparse
import re
import cssutils
import logging
cssutils.log.setLevel(logging.CRITICAL)
from tqdm import tqdm
from string import digits

import pdb

def load_json(fname):
    with open(fname, "r") as file:
        data = json.load(file)
    return data


def write_json(fname, data):
    with open(fname, "w") as file:
        json.dump(data, file, indent=4, separators=(",",":"), sort_keys=True)


def bb_relative_position(boxA, boxB):
    xA_c = (boxA[0]+boxA[2])/2
    yA_c = (boxA[1]+boxA[3])/2
    xB_c = (boxB[0]+boxB[2])/2
    yB_c = (boxB[1]+boxB[3])/2
    dist = np.sqrt((xA_c - xB_c)**2 + (yA_c - yB_c)**2)
    cosAB = (xA_c-xB_c) / dist
    sinAB = (yB_c-yA_c) / dist
    return cosAB, sinAB
    

def eval_spatial_relation(bbox1, bbox2):
    theta = np.sqrt(2)/2
    relation = 'diagonal'

    if bbox1 == bbox2:
        return relation
    
    cosine, sine = bb_relative_position(bbox1, bbox2)

    if cosine > theta:
        relation = 'right'
    elif sine > theta:
        relation = 'top'
    elif cosine < -theta:
        relation = 'left'
    elif sine < -theta:
        relation = 'bottom'
    
    return relation


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def parse_layout(string, canvas_size=64, no_integer=False):
    # Extract category: everything before the first '{'
    match_cat = re.search(r'^(.*?)\{', string)
    if not match_cat:
        # Fallback for lines without braces
        idx = string.find(' ')
        if idx == -1: return None, None
        category = string[:idx].strip()
        content = string[idx:].strip()
    else:
        category = match_cat.group(1).strip()
        # Extract content inside braces
        match_content = re.search(r'\{(.*?)\}', string)
        if not match_content:
            match_content = re.search(r'\{(.*)$', string)
        content = match_content.group(1) if match_content else ""

    parsed_category = re.sub(r'[0-9]', '', category.replace(' ', '-')).strip()
    
    # Use regex to find all key-value pairs
    # Supports "key: value", "key=value", optional "px", optional ";"
    pairs = re.findall(r'(\w+)\s*[:=]\s*([\d\.-]+)(?:px|degrees)?', content)
    bbox_dict = {k.lower().strip(): float(v) for k, v in pairs}

    required_keys = ['height', 'left', 'top', 'width']
    if not all(k in bbox_dict for k in required_keys):
        return None, None

    if no_integer:
        bbox = {k: bbox_dict[k] for k in required_keys}
    else:
        bbox = {k: int(bbox_dict[k]) for k in required_keys}

    # left, top, right, bottom
    bbox = [bbox['left'], bbox['top'], min(bbox['left']+bbox['width'], canvas_size), min(bbox['top']+bbox['height'], canvas_size)]
    if bbox[0] >= canvas_size or bbox[1] >= canvas_size:
        return None, None
    bbox = [float(b)/canvas_size for b in bbox]

    return parsed_category, bbox


def parse_3D_layout(string, unit='m'):
    # Extract category: everything before the first '{'
    match_cat = re.search(r'^(.*?)\{', string)
    if not match_cat:
        return None, None
    category = match_cat.group(1).strip()
    parsed_category = re.sub(r'[0-9]', '', category.replace(' ', '-')).strip()

    # Extract content inside braces
    match_content = re.search(r'\{(.*?)\}', string)
    if not match_content:
        match_content = re.search(r'\{(.*)$', string)
    content = match_content.group(1) if match_content else ""

    # Use regex to find all key-value pairs
    pairs = re.findall(r'(\w+)\s*[:=]\s*([\d\.-]+)(?:px|degrees|m)?', content)
    bbox_dict = {k.lower().strip(): float(v) for k, v in pairs}

    required_keys = ['height', 'width', 'length', 'orientation', 'left', 'top', 'depth']
    if not all(k in bbox_dict for k in required_keys):
        return None, None

    return parsed_category, bbox_dict



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--files", nargs="+")
    args = parser.parse_args()
    
    for fname in args.files:
        basename = os.path.basename(fname)
        dirname = os.path.dirname(fname)

        assert "raw" not in basename
        response = load_json(fname)

        print(f"Parsing {basename}")
        for r in tqdm(response):
            layout = r['text'].strip().strip("\n").strip().split("\n")
            assert len(layout) >= 2
            r['objects'] = []

            for elm in layout:
                selector_text, bbox = parse_layout(elm)
                if selector_text == None:
                    continue
                if sum(bbox) == 0:
                    print("Failed")
                r['objects'].append([selector_text, bbox])
        
        write_json(os.path.join(dirname, "parsed_"+basename), response)
                

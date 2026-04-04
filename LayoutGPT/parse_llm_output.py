import os
import argparse
import re
import cssutils
from tqdm import tqdm
from string import digits

from utils import *


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

    # x, y, w, h -> left, top, right, bottom
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
                

    
    
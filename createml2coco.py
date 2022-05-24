# This is the 1st version of the code that I developed and assumed YOLO format to be JSON.
# Only later I realized that this is code is written to convert CreateML format to COCO format.
# Left this code without deleting it.. This has nothing to do with PASCAL VOC to COCO format code.

from distutils.command.config import config
import json
import os
from pytesseract import pytesseract
from PIL import Image, ImageOps

def combine_annotations(temp_group):
    parent_node = temp_group.pop()

    if len(temp_group) == 0:
        temp_group.appen(parent_node)

    parent_node["text"] = (" ".join([anno["text"] for anno in temp_group]) if len(temp_group) > 0 else parent_node["text"])

    for anno in temp_group:
        parent_node["words"].append(
            {k:v for k,v in anno.items() if k in ("box", "text")}
        )
    return parent_node

def group_annotations(coco_annos):
    prev_group = None
    grouped_annos = []
    temp_group = []

    for i, anno in enumerate(coco_annos):
        if prev_group == None or prev_group == anno["label"]:
            temp_group.append(anno)
            prev_group = anno["label"]
        else:
            grouped_annos.append(combine_annotations(temp_group))
            temp_group = []
            prev_group = anno["label"]
            temp_group.append(anno)

        if i==len(coco_annos)-1:
            grouped_annos.append(combine_annotations(temp_group))

    for i in range(len(grouped_annos)):
        grouped_annos[i]["id"] = i

    return grouped_annos

pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
custom_config = r"--oem 3 --psm 6"

yolo_json_file = "vs_debug_error.json"
with open(yolo_json_file, "r") as fp:
    yolo_json = json.load(fp)

img_file = Image.open(yolo_json[0]["image"])

coco_annos = []

for i, item in enumerate(yolo_json[0]["annotations"]):
    coco_anno = dict()

    coords = item["coordinates"]
    x, y, width, height = coords["x"], coords["y"], coords["width"], coords["height"]
    box = [
        round(x - (width / 2)),
        round(y - (height / 2)),
        round(x + (width / 2)),
        round(y + (height / 2)),
    ]
    coco_anno["box"] = box

    img_part = ImageOps.grayscale(img_file.crop(box)) # Check this
    img_text = pytesseract.image_to_string(image=img_part, config=custom_config)
    img_text = img_text.rstrip("\n")
    coco_anno["text"] = img_text

    coco_anno["label"] = item["label"]
    coco_anno["words"] = []
    coco_anno["linking"] = []
    coco_anno["id"] = i + 1
    
    coco_annos.append(coco_anno)

grouped_coco_annos = group_annotations(coco_annos=coco_annos)

coco_data = {}
coco_data["form"] = grouped_coco_annos
json_data = json.dumps(coco_data)

folder, filename = os.path.split(yolo_json_file)
coco_filename = os.path.splitext(filename)[0] + "_coco.json"
coco_file_full_path = os.path.join(folder, coco_filename)

print(coco_file_full_path)
with open(coco_file_full_path, "w") as fp:
    fp.write(json_data)

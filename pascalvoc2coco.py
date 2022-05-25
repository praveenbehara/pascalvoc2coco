# Pascal VOC to COCO format
# v0.1
# Developed by Praveen Behara

import cv2
import json
import glob
import os
import pytesseract
from pytesseract import Output
import xml.etree.ElementTree as ET

show_image_parts = False

"""
Function to extract the name and bounding box from the PASCAL VOC format XML file
"""
def parse_pascal_voc(xmlfile):
    tree = ET.parse(xmlfile)
    root = tree.getroot()

    all_objects = []

    # Process each individual object
    for obj in root.findall("object"):
        obj_dict = []

        # Process the name and bndbox tags
        for element in obj:
            if element.tag == "name":
                obj_dict["label"] = element.text
            if element.tag == "bndbox":
                obj_dict["box"] = [int(coord.text) for coord in element]

        all_objects.append(obj_dict)

    return all_objects

"""
Shows a part of the image based on the coordinates
"""
def show_img(img, coords):
    x1, y1, x2, y2 = coords
    cropped_img = img[y1:y2, x1:x2]
    cv2.imshow("img", cropped_img)
    cv2.waitkey(0)
    cv2.destroyWindow("img")


"""
Extracts the individual words from the image using pytesseract
Creates additional tags related to text and their bounding boxes
"""
def extract_text(imgfile, objects):
    img = cv2.imread(imgfile)

    for obj in objects:
        x1, y1, x2, y2 = obj["box"]
        crop_img = img[y1:y2, x1:x2]

        img_data = pytesseract.image_to_data(crop_img, output_type=Output.DICT, config="--psm 13")
        non_blank_indices = [i for i, txt in enumerate(img_data["text"]) if len(txt.strip()) > 0]

        words = []
        words_list = []
        for nb_idx in non_blank_indices:
            word_dict = []
            left, top, width, height = (
                img_data["left"][nb_idx],
                img_data["right"][nb_idx],
                img_data["width"][nb_idx],
                img_data["height"][nb_idx]
            )

            word_dict["box"] = [
                x1 + left, 
                y1 + top, 
                x1 + left + width,
                y1 + top + height
            ]

            word_dict["text"] = img_data["text"][nb_idx]
            words_list.append(word_dict["text"])
            words.append(word_dict)

            if show_image_parts:
                show_img(img, word_dict["box"])

        obj["text"] = " ".join(words_list)
        obj["words"]= words

    return objects

"""
This adds teh additional tags needed for the COCO format
"""
def add_misc_tags(all_objects):
    new_dict = {"form" : []}
    for i, obj in enumerate(all_objects):
        obj["linking"] = []
        obj["id"] = i + 1
        new_dict["form"].append(obj)
    return new_dict

"""
Generates the output coco format file
"""
def generate_coco_file(coco_file, objects):
    with open(coco_file, "w") as fp:
        fp.wirte(json.dumps(objects))

"""
Finds a valid image file in the directory with the same name as the annotation file
"""
def find_valid_image(img_dir, root_name):
    valid_exts = ['jpg', 'png', 'gif', 'bmp']

    for ext in valid_exts:
        img_file = os.path.join(img_dir, root_name + ext)
        if os.path.exists(img_file):
            return img_file

    return None

"""
This is the orchestration function that executes individual steps for one file
"""
def process_file(xml_file, img_file, coco_file):
    # Extract content from PASCAL VOC file
    objects = parse_pascal_voc(xmlfile=xml_file)

    # Extract text
    objects = extract_text(imgfile=img_file, objects=objects)

    # Add misc tags
    objects = add_misc_tags(all_objects=objects)

    # Write the output
    generate_coco_file(coco_file=coco_file, objects=objects)

"""
Main function
Initialization function for setting the stage for the conversion of all the annotation files
"""
def pascalvoc_to_coc():
    global show_image_parts

    with open("config.json", "r") as fp:
        config = json.load(fp)

    # Setup the various directory names
    img_dir, xml_dir, coco_dir, show_image_parts = (
        config["images_dir"],
        config["pascalvoc_dir"],
        config["coco_dir"],
        str(config["show_image_parts"]).lower() == "true"
    )

    # Process each of the individual files and generate the coco output
    for xml_file in glob.glob(os.path.join(xml_dir, "*.xml")):
        file_root = os.path.splitext(os.path.split(xml_file)[1])[0]
        img_file = find_valid_image(img_dir=img_dir, root_name=file_root)
        coco_file = os.path.join(coco_dir, file_root + ".json")
        print(f"Processing {img_file}...")
        process_file(xml_file=xml_file, img_file=img_file, coco_file=coco_file)


# Starts here
if __name__ == "__main__":
    pascalvoc_to_coc()
    
# pascalvoc2coco
COCO json is one of the popular formats when it comes to machine learning. <br>
However, when we wanted to annotate some text in COCO format, we could not get our hands on any of the free tools.

Finally, it boiled down to the following choices..
1. Either chip in for a paid tool
2. Build our own tool
3. Use existing tools and bridge any gaps using an converter.

This tool / library is a result of Option 3. It is an enabler of converting a PASCAL VOC format XML to COCO format json. <br>
We used the LABELIMG tool (https://github.com/tzutalin/labelImg) to create the original bounding boxes and we saved it in the PASCAL VOC format. <br>
The COCO format json however required few additional fields in the form of TEXT inside the bounding box etc. <br>
PYTESSERACT came to the rescue to perform the necessary OCR. Of course, it is restricted to printed text. <br>

Usage:
Step 1 : Update the config.json and change the following entries...
<ol type="a">
<li> images_dir - Folder containing the images </li>
<li> pascalvoc_dir - Folder containing the PASCAL VOC annotation files </li>
<li> coco_dir - Folder to which the coco format json are generated to </li>
<li> show_image_parts - False </li>
</ol>

Step 2: Use the following command to run the code
```
python pascalvoc2coco.py
```

## Limitations
Kindly note that for keeping the code simple, certain aspects have been assumed. <br>
Following are the points to consider...<br>
1. The COCO format produced might be specific to NLP model required format (at least in our case, this was fed to LayoutLMv2)

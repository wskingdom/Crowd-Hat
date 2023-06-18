1a_generate_gt_custom_dots_boxes.py
generates custom binary ground truth dot maps with dilation maximum of (7, head width/2, head height/2) pixels and not touching nearest neighbor. 
By default output is in <datapath>/gt_map_custom2_boxes

1b_generate_gt_dotmap.py
Generates basic binary ground truth dot maps where each dot is just a single pixel with value = one. 
By default output is in <datapath>/ground-truth_dots

To run:
in main, set images_dir, json_dir, out_dir.
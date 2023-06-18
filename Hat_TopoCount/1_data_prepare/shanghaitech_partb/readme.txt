1a_generate_gt_custom_dots.py
generates custom binary ground truth dot maps with maximum dilation of 7 pixels and not touching nearest neighbor. 
By default output is in <datapath>/gt_map_custom2

1b_generate_gt_dotmap.py
Generates basic binary ground truth dot maps where each dot is just a single pixel with value = one. 
By default output is in <datapath>/ground-truth_dots

To run:
in main, set images_dir, mat_dir, out_dir for train and test datasets.
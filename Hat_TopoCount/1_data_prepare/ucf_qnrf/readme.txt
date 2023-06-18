1a_generate_gt_custom_dots.py
Generates custom binary ground truth dot maps with maximum dilation of 7 pixels and not touching nearest neighbor. 
By default output is in <datapath>/gt_map_custom2_scalelong2048 and <datapath>/img_scalelong2048 for train dataset
and in <datapath>/gt_map_custom2_scaleshort2048 and img_scaleshort2048 for test dataset.

1b_generate_gt_dotmap.py
Generates basic binary ground truth dot maps where each dot is just a single pixel with value = one. 
By default output is in <datapath>/ground-truth_dots_scalelong2048 and <datapath>/img_scalelong2048 for train dataset
and in <datapath>/ground-truth_dots_scaleshort2048 and img_scaleshort2048 for test dataset.

To run:
in main, set images_dir, mat_dir, out_dir, out_dir_img for train and test datasets.
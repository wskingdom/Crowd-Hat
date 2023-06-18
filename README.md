# Boosting Detection in Crowd Analysis via Underutilized Output Features

### [Project Page](https://fredfyyang.github.io/Crowd-Hat/) |   [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Boosting_Detection_in_Crowd_Analysis_via_Underutilized_Output_Features_CVPR_2023_paper.pdf)
<br>

<img src='imgs/pipeline.png' align="right" width=960>  
  

<br><br><br>
This repository contains the official PyTorch implementation of our paper [Boosting Detection in Crowd Analysis via Underutilized Output Features](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Boosting_Detection_in_Crowd_Analysis_via_Underutilized_Output_Features_CVPR_2023_paper.pdf). We provide a plug-and-play module leveraging the detection outputs as features to boost the performance in crowd analysis.


[Boosting Detection in Crowd Analysis via Underutilized Output Features](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Boosting_Detection_in_Crowd_Analysis_via_Underutilized_Output_Features_CVPR_2023_paper.pdf)  
 [Shaokai Wu*](), [Fengyu Yang*](https://fredfyyang.github.io/)<br>
Jilin University and University of Michigan<br>
 In CVPR 2023


## Code coming soon...

<!-- ## Environment
To setup the environment, please simply run

```bash
conda env create -f environment.yml
conda activate TDIS
```

## Touch and Go Dataset
Data can be downloaded from [Touch and Go Dataset](https://drive.google.com/drive/folders/1NDasyshDCL9aaQzxjn_-Q5MBURRT360B).

### Preprocessing
- Convert Video into frames (within the dataset):
```bash
cd touch_and_go
python extract_frame.py
```

- Sample frames and train/test split(within the TDIS code):
```bash
cd TDIS
python datasets/touch_and_go/generate_train_test.py  
```
We have already provided train/test split of our implementation in the `./datasets/touch_and_go/` folder.

## TDIS Training and Test
We provide training and evaluation scripts under `./scripts`, please check each bash file before running. Or you can run the code below:
- Training
```bash
python train.py --dataroot path/to/the/dataset --name touch_and_go --dataset_mode touch_and_go --model TDIS 
```
The checkpoints will be stored at `./checkpoints/touch_and_go/`

- Test the model
```bash
python test.py --dataroot path/to/the/dataset --name touch_and_go --dataset_mode touch_and_go --model TDIS
```
The test results will be saved to a html file at `./results/touch_and_go/test_latest/index.html` -->

### Citation
If you use this code for your research, please cite our [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Boosting_Detection_in_Crowd_Analysis_via_Underutilized_Output_Features_CVPR_2023_paper.pdf).
```
@InProceedings{Wu_2023_boosting,
      author    = {Wu, Shaokai and Yang, Fengyu},
      title     = {Boosting Detection in Crowd Analysis via Underutilized Output Features},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      month     = {June},
      year      = {2023},
      pages     = {15609-15618}
}
```

<!-- ### Acknowledgments
We thank Xiaofeng Guo and Yufan Zhang for the extensive help with the GelSight sensor, and thank Daniel Geng, Yuexi Du and Zhaoying Pan for the helpful discussions. This work was supported in part by Cisco Systems and Wang Chu Chien-Wen Research Scholarship. -->

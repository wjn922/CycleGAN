# CycleGAN
The repository is for the project of course COMP8503.

## Structure
```
|-- cyclegan_pytorch  # core code    
|-- demo              # demo folder     
|-- data              # folder for placing the datasets    
    |-- horse2zebra    
        |-- trainA   
        |-- trainB   
        |-- testA    
        |-- testB
    |-- ...
|-- weights           # folder for placing the model weights    
    |-- horse2zebra   
    |-- ...   
|-- outputs           # folder for placing the intermediate generated images during training   
|-- results           # folder for placing the generated images for test.py   
    |-- horse2zebra   
        |-- A   
        |-- B      
    |-- ...
|-- runs              # folder for tensorboard
|-- evaluation        # for evaluating the classification, storing code and resnet pretrained weight    
|-- scripts           # command line  
train.py      
test.py          
test_classification.py        
test_image.py     
test_video.py     
```

## Requirements
* CUDA>=9.2 (if build pytorch from source, CUDA9.0 is also available)
* Python>=3.6
* Pytorch>=1.3


## Install 
```
git clone https://github.com/wjn922/CycleGAN.git
cd CycleGAN

conda create -n visual python==3.7   
conda activate visual    
pip install torch==1.5.0 torchvision==0.6.0
pip install -r requirements.txt
```

## Datasets
In the projects, 8 datasets are used: **horse2zebra**, **apple2orange**, **summer2winter_yosemite**, **facades**, **cezanne2photo**, **monet2photo**, **ukiyoe2photo**, **vangogh2photo**. You can either download the .zip files from [official website](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/) or my [Goole drive](https://drive.google.com/drive/folders/1FDrQd2IwPUlezIQ1DEG6V-D_PoZzs_Q-?usp=sharing).      
After downloading the .zip file, please put it in the `data` folder and unzip it.

## Pretrained Model
We release the pretrained model in [Google drive](https://drive.google.com/drive/folders/1288wgHpT1aDfvcX9Xqxs4gr4VX1hsMBu?usp=sharing). You can download them and put them into the corresponding dataset folder under `weights` folder.  
You can also find the Resnet50 pretrained weight in the above link, downloading it and placing it in the `evaluation` folder for testing classification.

## Run
**NOTE**: Need adding `--cuda` to enable CUDA training and testing.
### 1. Train
```
python train.py --cuda --dataset [dataset_name]
```
### 2. Test
This is used for traslating all the images for a given dataset.
```
python test.py --dataset [dataset_name] --cuda --netG_A2B [pretrained netG_A2B name] --netG_B2A [pretrained netG_B2A name]
```
### 3. Test Classification
This is used for evaluating the classification accuracy for 2 datasets: **horse2zebra**, **apple2orange**.
```
python test_classification.py --cuda --datapath [/path/to/dataset] --label [label_name] --batch_size 32
```
For example
```
python test_classification.py --cuda --datapath ./data/horse2zebra/testA --label horse --batch_size 32
```
### 4. Test Image
This is used for translating a single image.
```
python test_image.py --cuda --file [/path/to/image] --model-name [/path/to/pretrained weight]
```
For example
```
python test_image.py --cuda --file demo/horse.png --model-name weights/horse2zebra/netG_A2B_epoch_199.pth
```

### 5. Test Video
This is used for translating a single image.
```
python test_video.py --cuda --file [/path/to/video] --model-name [/path/to/pretrained weight]
```
For example
```
python test_video.py --cuda --file demo/horse.mp4 --model-name weights/horse2zebra/netG_A2B_epoch_199.pth
```

## Demo

## Citation
Thanks to the repository [https://github.com/Lornatang/CycleGAN-PyTorch](https://github.com/Lornatang/CycleGAN-PyTorch) for reference.

```
@inproceedings{zhu2017unpaired,
  title={Unpaired image-to-image translation using cycle-consistent adversarial networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={2223--2232},
  year={2017}
}
```

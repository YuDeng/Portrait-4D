# Portrait4D: A One-Shot Video-Driven 4D Head Synthesizer

<p align="center"> 
<img src="/assets/teaser.jpg">
</p>

This repository contains the official pytorch implementation of the following papers:

> [**Portrait4D: Learning One-Shot 4D Head Avatar Synthesis using Synthetic Data**](https://yudeng.github.io/Portrait4D/), CVPR 2024 </br>
> Yu Deng, Duomin Wang, Xiaohang Ren, Xingyu Chen, Baoyuan Wang </br>
> Xiaobing.AI

> [**Portrait4D-v2: Pseudo Multi-View Data Creates Better 4D Head Synthesizer**](https://yudeng.github.io/Portrait4D-v2/), arXiv 2024 </br>
> Yu Deng, Duomin Wang, Baoyuan Wang </br>
> Xiaobing.AI

## Abstract
Portrait4D: Learning One-Shot 4D Head Avatar Synthesis using Synthetic Data
> Existing one-shot 4D head synthesis methods usually learn from monocular videos with the aid of 3DMM reconstruction, yet the latter is evenly challenging which restricts them from reasonable 4D head synthesis. We present a method to learn one-shot 4D head synthesis via large-scale synthetic data. The key is to first learn a part-wise 4D generative model from monocular images via adversarial learning, to synthesize multi-view images of diverse identities and full motions as training data; then leverage a transformer-based animatable triplane reconstructor to learn 4D head reconstruction using the synthetic data. A novel learning strategy is enforced to enhance the generalizability to real images by disentangling the learning process of 3D reconstruction and reenactment. Experiments demonstrate our superiority over the prior art.

Portrait4D-v2: Pseudo Multi-View Data Creates Better 4D Head Synthesizer
> In this paper, we propose a novel learning approach for feed-forward one-shot 4D head avatar synthesis. Different from existing methods that often learn from reconstructing monocular videos guided by 3DMM, we employ pseudo multi-view videos to learn a 4D head synthesizer in a data-driven manner, avoiding reliance on inaccurate 3DMM reconstruction that could be detrimental to the synthesis performance. The key idea is to first learn a 3D head synthesizer using synthetic multi-view images to convert monocular real videos into multi-view ones, and then utilize the pseudo multi-view videos to learn a 4D head synthesizer via cross-view self-reenactment. By leveraging a simple vision transformer backbone with motion-aware cross-attentions, our method exhibits superior performance compared to previous methods in terms of reconstruction fidelity, geometry consistency, and motion control accuracy. We hope our method offers novel insights into integrating 3D priors with 2D supervisions for improved 4D head avatar creation.

## Requirements
- The code is only tested on Linux.
- Python>=3.8 and PyTorch>=1.11.0.
- CUDA toolkit 11.3 or later.
- One or more high-end NVIDIA GPUs with NVIDIA drivers installed. For training and finetuning, we recommend using 8 Tesla A100 GPUs with 80 GB memory. For inference, a V100 GPU with 32 GB memory is enough.
- [Pytorch3d](https://github.com/facebookresearch/pytorch3d) and other python libraries. See requirements.txt for full library dependencies.

## Installation
<details>
<summary><span >Install step by step</span></summary>


Clone the repository and create a conda environment:

```
git clone https://github.com/YuDeng/Portrait-4D.git
cd Portrait-4D
conda create -n portrait4d python=3.8
conda activate portrait4d
```
Install required python libraries:
```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113
```
Install Pytorch3d from source (see [here](https://github.com/facebookresearch/pytorch3d/blob/v0.6.2/INSTALL.md) for details):
```
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.6.2"
```

</details>

## Quick Inference

<details>
<summary><span >Download prequisite models</span></summary>

Our method utilizes [FLAME head model](https://github.com/Rubikplayer/flame-fitting) for head pose control (as well as shape and expression control in GenHead). We provide a quick access to the model on [HuggingFace](https://huggingface.co/bEijuuu/Portrait4D/tree/main/models/FLAME). 

We also rely on [PD-FGC](https://github.com/Dorniwang/PD-FGC-inference) to extract motion embeddings for Portrait4D and Portrait4D-v2. The [pretrained weights](https://huggingface.co/bEijuuu/Portrait4D/tree/main/models/pdfgc) can be found here.

Please download the above files into the following structure:
```
Portrait-4D/
│
└─── portrait4d/
    |
    └─── models/
        |
        └─── FLAME/
        |   |
        |   └─── geometry/
        |   |
        |   └─── mask/
        |
        └─── pdfgc/
            |
            └─── weights/
                |
                └─── motion_model.pth
```

Please also download the pre-trained weights of GenHead, Portrait4D and Portrait4D-v2 and put them into ./portrait4d/pretrained_models/:
|File (HuggingFace) |Description|
|:----:|:-----------:|
|[genhead-ffhq512](https://huggingface.co/bEijuuu/Portrait4D/blob/main/genhead-ffhq512.pkl) | Animatable 3D head generator conditioned on FLAME parameters, trained on FFHQ at 512x512 |
|[portrait4d-genhead512](https://huggingface.co/bEijuuu/Portrait4D/blob/main/portrait4d-genhead512.pkl) | One-shot video-driven 4D head synthesizer learned in Portrait4D |
|[portrait4d-static-genhead512](https://huggingface.co/bEijuuu/Portrait4D/blob/main/portrait4d-static-genhead512.pkl)| One-shot static 3D head synthesizer learned in Portrait4D-v2 |
|[portrait4d-v2-vfhq512](https://huggingface.co/bEijuuu/Portrait4D/blob/main/portrait4d-v2-vfhq512.pkl)| One-shot video-driven 4D head synthesizer learned in Portrait4D-v2 |

</details>

<details>
<summary><span >Generate images via pre-trained GenHead</span></summary>

1. GenHead requires FLAME parameters as extra conditions for controllable head image synthesis. We provide [pre-extracted FLAME parameters](https://huggingface.co/bEijuuu/Portrait4D/tree/main/data) from FFHQ and VFHQ datasets as a sampling set and please download them into the following structure:
```
Portrait-4D/
│
└─── portrait4d/
    |
    └─── data/
        |
        └─── ffhq_all_shape_n_c_params.npy
        |
        └─── vfhq_all_shape_n_c_params.npy
        |
        └─── ffhq_all_motion_params.npy
        |
        └─── vfhq_all_motion_params.npy
```  
2. Run the following command for arbitrary head image synthesis:
```
cd portrait4d
python gen_images_genhead.py --network=./pretrained_models/genhead-ffhq512.pkl --outdir=<custom_resultdir>
```

</details>

<details>
<summary><span >Synthesize 4D head avatars via pre-trained Portrait4D/Portrait4D-v2</span></summary>

Run the following command for head reenactment via Portrait4D/Portrait4D-v2:
```
cd portrait4d
# Use Portrait4D
python gen_images_portrait4d.py --network=./pretrained_models/portrait4d-genhead512.pkl --outdir=<custom_resultdir>

# Use Portrait4D-v2
python gen_images_portrait4d.py --network=./pretrained_models/portrait4d-v2-vfhq512.pkl --outdir=<custom_resultdir>

# Extract shapes via marching cubes
python gen_images_portrait4d.py --network=./pretrained_models/portrait4d-v2-vfhq512.pkl --outdir=<custom_resultdir> --shape=1

```
By default, it will generate reenacted images given sources and drivings from ./portrait4d/examples. 

To generate reenacted results with custom sources and drivings, please follow the data preprocessing instruction below.

</details>

## Data Preprocessing for Custom Images

<details>
<summary><span >Landmark detection, 3D face reconstruction, and cropping</span></summary>

We use [RetinaFace](https://github.com/serengil/retinaface) for face detection, [H3R](https://github.com/baoshengyu/H3R) and [3DFAN](https://github.com/1adrianb/2D-and-3D-face-alignment) for 2D and 3D landmark prediction, respectively, and [Deep3DFace](https://github.com/sicxu/Deep3DFaceRecon_pytorch) for 3D face reconstruction.

We provide a quick access to these pre-trained models:
   
|Model|Download link (Huggingface)|
|:----:|:-----------:|
| RetinaFace | https://huggingface.co/bEijuuu/Portrait4D/tree/main/data_preprocess/assets/facedetect  |
| H3R | https://huggingface.co/bEijuuu/Portrait4D/tree/main/data_preprocess/assets/hrnet_w18_wflw |
| 3DFAN | https://huggingface.co/bEijuuu/Portrait4D/tree/main/data_preprocess/assets/landmark3d |
| Deep3DFace | https://huggingface.co/bEijuuu/Portrait4D/tree/main/data_preprocess/assets/facerecon |

1. Please download the above models and organize them as follows:
```
Portrait-4D/
│
└─── data_preprocess/
    |
    └─── assets/
        |
        └─── facedetect/
        |   |
        |   └─── retinaface/
        |       |
        |       └─── Resnet50_Final.pth
        |
        └─── facerecon/
        |   |
        |   └─── deep3d_2023/
        |   |   |
        |   |   └─── epoch_20.pth
        |   |
        |   └─── bfm/
        |       |
        |       └─── BFM_model.mat
        |       |
        |       └─── BFM_model_front.mat
        |
        └─── hrnet_w18_wflw/
        |   |
        |   └─── h3r/
        |       |
        |       └─── model.pth
        |   
        └─── landmark3d/
            |
            └─── 3DFAN4-4a694010b9.zip
            |   
            └─── depth-6c4283c0e0.zip

```
2. Run the following command for landmark detection, 3D face reconstruction, and image cropping:
```
cd data_preprocess
python preprocess_dir.py --input_dir=<custom_indir> --save_dir=<custom_outdir>
```
The results should be stored in the following structure:
```
<custom_outdir>/
│
└─── align_images/
|   |
|   └─── *.<image_extension>  # aligned images at 512x512 resolution
|
└─── 2dldmks_align/
|   |
|   └─── *.npy  # extracted 98 2d landmarks
|
└─── 3dldmks_align/
|   |
|   └─── *.npy  # extracted 68 3d landmarks
|
└─── bfm_params/
|   |
|   └─── *.npy  # reconstructed 3d face parameters
|
└─── bfm_vis/
    |
    └─── *.<image_extension>  # visualizations of the reconstructed 3d face
```

</details>

<details>
<summary><span >BFM to FLAME parameter transformation</span></summary>

Since our method relies on FLAME model for head pose control, it is required to transfer the obtained BFM parameters in <custom_outdir> to FLAME ones.

#### For quick inference, we provide a simplified transformation process which utilizes a light-weight MLP to map the BFM parameters to the FLAME space:
```
cd data_preprocess
python bfm2flame_simplified.py --input_dir=<custom_outdir> --save_dir=<custom_outdir>
```

The obtained FLAME parameters will be saved in:
```
<custom_outdir>/
│
└─── bfm2flame_params_simplified/
    |
    └─── *.npy  # FLAME parameters mapped from bfm_params via a light-weight MLP
```
You can then run head reenactment with the custom images via:
```
cd portrait4d

# Use Portrait4D-v2
python gen_images_portrait4d.py --network=./pretrained_models/portrait4d-v2-vfhq512.pkl \
	--srcdir=<custom_outdir> \
	--tardir=<custom_outdir> \
	--outdir=<custom_resultdir> \
	--use_simplified=1
```

#### Alternatively, we provide a full BFM-to-FLAME transformation process which we use during our training and inference:
For the full process, we first use [BFM_to_FLAME](https://github.com/TimoBolkart/BFM_to_FLAME/tree/main) to convert BFM parameters to FLAME ones via mesh-based optimization; then, we conduct landmark-based optimization to refine the FLAME parameters.

1. BFM_to_FLAME relies on [MPI-IS/mesh](https://github.com/MPI-IS/mesh) library to run. Clone the library via:
```
git clone https://github.com/MPI-IS/mesh.git
```
Before installation, remove all dependencies in ./mesh/requirements.txt (otherwise it will conflict with the requirements.txt of our repository). Also replace the "--install-option" option with "--config-settings" in ./mesh/Makefile (the former is out-of-date). After that, run the following commands to install the library:
```
sudo apt-get install libboost-dev

cd mesh
BOOST_INCLUDE_DIRS=/path/to/boost/include make all
```

2. Then, download [required models](https://huggingface.co/bEijuuu/Portrait4D/tree/main/data_preprocess/bfm_to_flame) of BFM-to-FLAME into the following structure:
```
Portrait-4D/
│
└─── data_preprocess/
    |
    └─── bfm_to_flame/
        |
        └─── data/
        |
        └─── model/
```
3. Run the following command to launch BFM-to-FLAME transformation:
```
cd data_preprocess/bfm_to_flame
python run_convert.py --input_dir=<custom_outdir> --save_dir=<custom_outdir> --n_thread=<number_of_thread> --instance_per_thread=<instance_per_thread>
```

The script conducts multi-thread mesh-based optimization on CPUs and saves the results in:
```
<custom_outdir>/
│
└─── bfm2flame_params/
    |
    └─── *.npy  # FLAME parameters obtained via mesh-based optimization
```

4. Finally, run the following script to conduct landmark-based optimization:
```
cd data_preprocess
python flame_optim_batch_singleframe.py --data_dir=<custom_outdir> --save_dir=<custom_outdir> --batchsize=<batchsize>
```
The optimized flame parameters will be saved into:
```
<custom_outdir>/
│
└─── flame_optim_params/
    |
    └─── *.npy  # FLAME parameters obtained via landmark-based optimization
```

You can then run head reenactment with the custom images via:
```
cd portrait4d

# Use Portrait4D-v2
python gen_images_portrait4d.py --network=./pretrained_models/portrait4d-v2-vfhq512.pkl \
	--srcdir=<custom_outdir> \
	--tardir=<custom_outdir> \
	--outdir=<custom_resultdir>
```

5. If you want to optimize FLAME parameters for consecutive frames extracted from a same video clip, you can also run the following script:
```
cd data_preprocess
python flame_optim_batch_multiframe.py --data_dir=<parentdir_of_custom_outdir> --save_dir=<parentdir_of_custom_outdir>
```
The above script will optimize for each <custom_outdir> in the <parentdir_of_custom_outdir>, where <custom_outdir> contains consecutive frames of a video clip. During optimization, the shape parameters of different frames in <custom_outdir> are forced to be identical.

</details>

<details>
<summary><span >Face segmentation and motion embedding extraction (for training only)</span></summary>

For training, we also extract segmentation masks and PD-FGC motion embeddings for each image. 

For face segmenation, we suggest using [Mask2Former](https://github.com/facebookresearch/Mask2Former).

For motion embedding extraction, please run the following script:
```
cd data_preprocess
python extract_pdfgc.py --input_dir=<custom_outdir> --save_dir=<custom_outdir>
```
where <custom_outdir> stores the extracted 2D landmarks and cropped images as described in the landmark detection section.

</details>

## Training from Scratch

<details>
<summary><span >Download prequisite models for loss computation</span></summary>

Learning portrait4D and portrait4D-v2 requires [LPIPS](https://github.com/richzhang/PerceptualSimilarity) and [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) for loss computation.

The checkpoints of these models can be found here:
|Model|Download link (Huggingface)|
|:----:|:-----------:|
| LPIPS | https://huggingface.co/bEijuuu/Portrait4D/tree/main/models/lpips |
| ArcFace | https://huggingface.co/bEijuuu/Portrait4D/tree/main/models/arcface |

Please download the pre-trained weights and save them into:
```
Portrait-4D/
│
└─── portrait4d/
    |
    └─── models/
        |
        └─── arcface/
        |   |
        |   └─── ms1mv3_arcface_r18_fp16/
        |   	|
        |   	└─── backbone.pth
        |
        └─── lpips/
            |
            └─── weights/
		|
		└─── v0.1/
		    |
		    └─── alex.pth
		    |
		    └─── vgg.pth
```

</details>

<details>
<summary><span >Getting started with toy datasets</span></summary>

We provide two toy datasets in [LMDB](https://lmdb.readthedocs.io/en/release/) format for demonstration.

Download the [toy datasets](https://huggingface.co/bEijuuu/Portrait4D/tree/main/data) and store them in the following structure:
```
Portrait-4D/
│
└─── portrait4d/
    |
    └─── data/
	    |
	    └─── FFHQ_512_50/
	    |
	    └─── VFHQ_sub50_512_4/
```
"FFHQ_512_50" is used for learning GenHead, which contains images of 50 FFHQ subjects at 512x512 resolution; "VFHQ_sub50_512_4" is used for learning Portrait4D-v2, which contains video clips of 4 VFHQ subjects at 512x512 resolution and each video clip contains 50 random frames. Note that Portrait4D-v1 learns from synthetic data of GenHead which does not require extra training data.

#### For GenHead training, run:
```
cd portrait4d
python train_genhead.py --cfg=./configs/genhead-ffhq512-toy.yaml
```
By default, experiment results will be saved in ./training-runs-genhead/

#### To train Portrait4D-v1, run:
```
cd portrait4d
python train_recon_v1.py --cfg=./configs/portrait4d-genhead512.yaml

# Static 3D model used for Portrait4D-v2
python train_recon_v1.py --cfg=./configs/portrait4d-static-genhead512.yaml
```
By default, experiment results will be saved in ./training-runs-portrait4d/.

#### To train Portrait4D-v2, run:
```
cd portrait4d
python train_recon_v2.py --cfg=./configs/portrait4d-v2-vfhq512-toy.yaml
```
By default, experiment results will be saved in ./training-runs-portrait4d-v2/. 
</details>

<details>
<summary><span >Creating full LMDB datasets for training</span></summary>

1. Please first download in-the-wild images of [FFHQ](https://drive.google.com/drive/folders/1ZX7QOy6LZuTLTnsOtQk-kmKq2-69l5hu) and [VFHQ](https://liangbinxie.github.io/projects/vfhq/) datasets.

2. Then, follow the instruction in the data preprocessing section to obtain landmarks, cropped images, reconstructed flame parameters, segmentations, and motion embeddings for each image. The processed datasets should be organized as follows:
```
# For FFHQ dataset:
<ffhq_root_dir>/
│
└─── 00000/
|    │
|    └─── align_images/*.png # aligned images at 512x512 resolution
|    |
|    └─── flame_optim_params/*.npy  # FLAME parameters of each aligned image
|    │
|    └─── segs/*.png  # segmentation mask of each aligned image
|    |
|    └─── motion_feats/*.npy  # pd-fgc motion embedding of each aligned image
|
└─── 01000/
|
└─── 02000/
|
└─── ...

# For VFHQ dataset:
<vfhq_root_dir>/
│
└─── <subject1>/
|    │
|    └─── align_images/
|    |
|    └─── flame_optim_params/
|    |
|    └─── segs/
|    |
|    └─── motion_feats/
|
└─── <subject2>/
|
└─── <subject3>/
|
└─── ...
```
For FFHQ, each subfolder contains 1000 images following the original data structure [here](https://drive.google.com/drive/folders/1ZX7QOy6LZuTLTnsOtQk-kmKq2-69l5hu). For example, 00000/ contains images indexed from 0 to 999. We also horizontally flip the 70000 images in FFHQ and store their processed results into successive subfolders range from 70000/ to 139000/.

For VFHQ, each subfolder stores processed results from one video clip.

3. Finally, generate the LMDB files for each dataset via the following scripts:
```
# For FFHQ
cd generate_lmdb/ffhq
python generate_lmdb.py --data_dir=<ffhq_root_dir>

# For VFHQ
cd generate_lmdb/vfhq
python generate_lmdb.py --data_dir=<vfhq_root_dir>

```
The scripts will save the files into ./portrait4d/data for training.

To reproduce the results in our paper, we recommand using at least 4 A100 GPUs with 40GB memory and a total batchsize of 32 for training GenHead; For training portrait4D and portrait4D-v2, we recommand using 8 A100 GPUs with 80GB memory and a total batchsize of 32 (a batchsize of 32 causes OOM error on 8 40GB A100 GPUs in our experiments).

</details>

## Evaluation
Run the following script to calculate FID metric for GenHead:
```
cd portrait4d
python calc_metrics_deform.py --network=<path_to_genhead.pkl> --data=<path_to_lmdb_folder>
```

## Citation

Please cite the following papers if this work helps your research:

    @inproceedings{deng2024portrait4d,
		title={Portrait4D: Learning One-Shot 4D Head Avatar Synthesis using Synthetic Data},
		author={Deng, Yu and Wang, Duomin and Ren, Xiaohang and Chen, Xingyu and Wang, Baoyuan},
		booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
		year={2024}
	}

    @article{deng2024portrait4dv2,
	      title={Portrait4D-v2: Pseudo Multi-View Data Creates Better 4D Head Synthesizer},
	      author={Deng, Yu and Wang, Duomin and Wang, Baoyuan},
	      journal={arXiv preprint arXiv:2403.13570},
	      year={2024}
    }

## Acknowledgements
We thank the following excellent open resources which make this project possible:
- [Pytorch_FLAME](https://github.com/soubhiksanyal/FLAME_PyTorch) and [DecaFLAME](https://github.com/yfeng95/DECA) for the FLAME model.
- [EG3D](https://github.com/NVlabs/eg3d) for the learning architecture.
- [Pytorch3D](https://github.com/facebookresearch/pytorch3d) for mesh processing and rendering.
- [PD-FGC](https://github.com/Dorniwang/PD-FGC-inference) for motion embedding extraction.
- [SegFormer](https://github.com/NVlabs/SegFormer) for backbone design of Portrait4D.
- [RetinaFace](https://github.com/serengil/retinaface), [H3R](https://github.com/baoshengyu/H3R), [3DFAN](https://github.com/1adrianb/2D-and-3D-face-alignment), [Deep3DFace](https://github.com/sicxu/Deep3DFaceRecon_pytorch), [BFM_to_FLAME](https://github.com/TimoBolkart/BFM_to_FLAME/tree/main), [Mask2Former](https://github.com/facebookresearch/Mask2Former) for face data preprocessing.
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity) and [InsightFace](https://github.com/deepinsight/insightface/tree/master) for loss computation.

## Related Works
Here are some related projects for one-shot 4D head avatars:
- [GPAvatar](https://github.com/xg-chu/GPAvatar)
- [Real3DPortrait](https://github.com/yerfor/Real3DPortrait)
- [GOHA](https://github.com/NVlabs/GOHA)
- [ROME](https://github.com/SamsungLabs/rome)





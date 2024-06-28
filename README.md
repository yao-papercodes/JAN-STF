# Joint Adversarial Network With Semantic and Topology Fusion for Cross-Scene Hyperspectral Image Classification
Implementation of papers:
- [Joint Adversarial Network With Semantic and Topology Fusion for Cross-Scene Hyperspectral Image Classification](https://ieeexplore.ieee.org/abstract/document/10559841)  
  IEEE Transactions on Geoscience and Remote Sensing (IEEE TGRS), 2024.  
  Ronghua Shang, Yuhao Xie, Weitong Zhang, Jie Feng, Songhua Xu

<div align=center>
	<img src="./figures/framework.png"/>
</div>

## Environment
Ubuntu 20.04.2 LTS, python 3.8.10, PyTorch 1.12.1.
## Datasets
Application website： [Houston, HyRANK, Pavia](https://github.com/YuxiangZhang-BIT/Data-CSHSI)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="./figures/Houston.png" width = "65%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Houston
  	</div>
</center>


<div align=center>
	<img src="./figures/HyRANK.png"/>
</div>
<div align=center>
	<img src="./figures/Pavia.png"/>
</div>
## Usage
```bash
cd code
bash TrainOnSourceDomain.sh     # First step
bash TransferToTargetDomain.sh  # Second step
```
## Citation
If you find our paper or code helpful, please cite our work.
```bash
@ARTICLE{10404024,
  author={Gao, Yuefang and Xie, Yuhao and Hu, Zeke Zexi and Chen, Tianshui and Lin, Liang},
  journal={IEEE Transactions on Multimedia}, 
  title={Adaptive Global-Local Representation Learning and Selection for Cross-Domain Facial Expression Recognition}, 
  year={2024},
  volume={26},
  number={},
  pages={6676-6688},
  keywords={Feature extraction;Adaptation models;Adversarial machine learning;Face recognition;Semantics;Data models;Representation learning;Domain adaptation;adverserial learning;Pseudo label generation;Facial expression recognition},
  doi={10.1109/TMM.2024.3355637}
}

@INPROCEEDINGS{9956069,
  author={Xie, Yuhao and Gao, Yuefang and Lin, Jiantao and Chen, Tianshui},
  booktitle={2022 26th International Conference on Pattern Recognition (ICPR)}, 
  title={Learning Consistent Global-Local Representation for Cross-Domain Facial Expression Recognition}, 
  year={2022},
  volume={},
  number={},
  pages={2489-2495},
  doi={10.1109/ICPR56361.2022.9956069}
}
```
## Contributors
For any questions, feel free to open an issue or contact us:
- <a href="mailto:yaoxie1001@gmail.com">yaoxie1001@gmail.com</a>

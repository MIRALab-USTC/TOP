# Accurate and Scalable Graph Neural Networks via Message Invariance

This is the code of paper 
**Accurate and Scalable Graph Neural Networks via Message Invariance**. 
Zhihao Shi, Jie Wang*, Zhiwei Zhuang, Xize Liang, Bin Li, Feng Wu. ICLR 2023. [[arXiv](https://arxiv.org/abs/2502.19693)]
[[ICLR-Official](https://openreview.net/forum?id=UqrFPhcmFp)]

## Dependencies

- Python 3.9
- PyTorch 2.4.0
- torch-geometric 2.5.3
- ogb 1.3.6
- hydra-core 1.3.2


## Python environment setup with Conda


```bash
conda create -n top python=3.9
conda activate top

conda install -c conda-forge cuda-toolkit=12.4

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

pip install torch_geometric==2.5.3
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html

pip install hydra-core

pip install -e .
```

### Running TOP

```bash
conda activate top
bash main.sh
```


## Citation
If you find this code useful, please consider citing the following papers.
```
@inproceedings{
shi2025accurate,
title={Accurate and Scalable Graph Neural Networks via Message Invariance},
author={Zhihao Shi and Jie Wang and Zhiwei Zhuang and Xize Liang and Bin Li and Feng Wu},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=UqrFPhcmFp}
}

@inproceedings{
shi2023lmc,
title={{LMC}: Fast Training of {GNN}s via Subgraph Sampling with Provable Convergence},
author={Zhihao Shi and Xize Liang and Jie Wang},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=5VBBA91N6n}
}
```

## Acknowledgement
We refer to the code of [PyGAS](https://github.com/rusty1s/pyg_autoscale). Thanks for their contributions.






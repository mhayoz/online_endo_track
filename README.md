<h1 align="center">

Online 3D reconstruction and dense tracking in endoscopic videos

[Paper](https://arxiv.org/abs/todo)

|                         Tracking                         |                 3D Semantic Segmentation                 |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| <img src="docs/P3_1_tracking.gif" style="zoom: 100%;" /> | <img src="docs/P3_1_semantic.gif" style="zoom: 100%;" /> |

</h1>

<br>

## Installation

```bash
pip install -r requirements.txt
git submodule update --init --recursive
pip install -e src/submodules/gaussian-rasterization
pip install -e src/submodules/simple-knn
```

## Run
Download the data from [StereoMIS Tracking](10.5281/zenodo.10867949) and unpack it in the repository base folder.
Once the data is downloaded, you can run our method with:
```bash
python run.py configs/StereoMIS/P3_1.yaml --log P3_1 --visualize
```

## Citing
If you find our work useful, please consider citing:

```BibTeX
@inproceedings{hayoz2024,
  author = {Hayoz, Michel and Hahne, Christopher and Kurmann, Thomas and Allan, Max and Beldi, Guido and Candinas, Daniel and Marquez-Neila, Pablo and Sznitman, Raphael},
  title = {Online 3D reconstruction and dense tracking in endoscopic videos}
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year = {2024},
  
}
```
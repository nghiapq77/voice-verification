# Zalo AI Challenge - Voice Verification
This repository contains the framework for training speaker verification model described in [2]  
with score normalization post-processing described in [3].

## Dependencies
```
pip install -r requirements.txt
```

## Data Preparation
1. Download the [public dataset](https://dl.challenge.zalo.ai/voice-verification/data/Train-Test-Data_v2.zip)
then put the training speakers data in `dataset/wavs` and public-test folder in `dataset/public-test`

2. Convert data (this will overwrite original data)
```python
python dataprep.py --save_path dataset/wavs --convert
```

3. Prepare the augment data  
```python
python dataprep.py --save_path dataset --augment
```

3. Generate train, validate list
```python
python dataprep.py --save_path dataset/wavs --generate --split_ratio -1
```

In addition to the Python dependencies, `wget` and `ffmpeg` must be installed on the system.

## Pretrained models
Pretrained models and corresponding cohorts can be downloaded from [here](https://drive.google.com/drive/folders/15FYmgHGKlF_JSyPGKfJzBRhQpBY5JcBw?usp=sharing).

## Training
Training from pretrained with augmentation for 500 epochs.
```python
python train.py --augment --max_epoch 500 --batch_size 320 --initial_model checkpoints/baseline_v2_ap.model
```

## Inference
1. Prepare cohorts
```python
python inference.py --prepare --save_path checkpoints/cohorts_final_500_f100.npy --initial_model checkpoints/final_500.model
```

2. Evaluate and tune thresholds
```python
python inference.py --eval --cohorts_path checkpoints/cohorts_final_500_f100.npy --initial_model checkpoints/final_500.model
```

3. Run on test set
```python
python inference.py --test --cohorts_path checkpoints/cohorts_final_500_f100.npy --test_threshold 1.7206447124481201 --test_path dataset --initial_model checkpoints/final_500.model
```

## Citation

[1] _In defence of metric learning for speaker recognition_
```
@inproceedings{chung2020in,
    title={In defence of metric learning for speaker recognition},
    author={Chung, Joon Son and Huh, Jaesung and Mun, Seongkyu and Lee, Minjae and Heo, Hee Soo and Choe, Soyeon and Ham, Chiheon and Jung, Sunghwan and Lee, Bong-Jin and Han, Icksang},
    booktitle={Interspeech},
    year={2020}
}
```

[2] _Clova baseline system for the VoxCeleb Speaker Recognition Challenge 2020_
```
@article{heo2020clova,
    title={Clova baseline system for the {VoxCeleb} Speaker Recognition Challenge 2020},
    author={Heo, Hee Soo and Lee, Bong-Jin and Huh, Jaesung and Chung, Joon Son},
    journal={arXiv preprint arXiv:2009.14153},
    year={2020}
}
```

[3] _Analysis of score normalization in multilingual speaker recognition_
```
@inproceedings{inproceedings,
    title = {Analysis of Score Normalization in Multilingual Speaker Recognition},
    author = {Matejka, Pavel and Novotny, Ondrej and Plchot, Oldřich and Burget, Lukas and Diez, Mireia and Černocký, Jan},
    booktitle = {Interspeech},
    year = {2017}
}
```

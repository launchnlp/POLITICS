# POLITICS
This repository contains code, data, and models for the NAACL 2022 findings paper [POLITICS: Pretraining with Same-story Article Comparison for Ideology Prediction and Stance Detection](https://aclanthology.org/2022.findings-naacl.101/).

<i>We are still refactoring and cleaning the downstream evaluation code, please stay tuned and check back later!</i>

## What's in this repo?
- Continue pretrained POLITICS model is available on [Huggingface](https://huggingface.co/launch/POLITICS) with model card ```launch/POLITICS```.
- Cleaned BIGNEWS, BIGNEWSBLN, and BIGNEWSALIGN are available to download after you fill out [this form](https://forms.gle/uBMoMMoySyV6HcJz5).
- Code for continued pretraining.

## Continued pretraining
To retrain POLITICS, simply run ```pretrain.sh```. You need to download the precessed data that contains the indices for entities and sentiment words from [this form](https://forms.gle/uBMoMMoySyV6HcJz5). After downloading, please move the data files and lexicon directory to the ```DATA_DIR``` defined in ```pretrain.sh```.

## Downstream tasks evaluation

Macro F1 score for all tasks are shown in the following table.
|                 | YT (cmt.) | CongS | HP    | AllS  | YT (user) | TW    | Ideo. avg | SEval (seen) | SEval (unseen) | Basil (sent.) | VAST  | Basil (art.) | Stan. avg | All avg |
|-----------------|-----------|-------|-------|-------|-----------|-------|-----------|--------------|----------------|---------------|-------|--------------|-----------|---------|
| BERT (base)     | 64.64     | 65.88 | 48.42 | 60.88 | 65.24     | 44.20 | 58.21     | 65.07        | 40.39          | 62.81         | 70.53 | 45.61        | 56.88     | 57.61   |
| RoBERTa (base)  | 66.72     | 67.25 | 60.43 | 74.75 | 67.98     | 48.90 | 64.34     | **70.15**        | **63.08**          | 68.16         | 76.25 | 41.36        | 63.80     | 64.09   |
| POLITICS (base) | **67.83**     | **70.86** | **70.25** | **74.93** | **78.73**     | **48.92** | **68.59**     | 69.41        | 61.26          | **73.41**         | **76.73** | **51.94**        | **66.55**     | **67.66**   |


## License
POLITICS is shared under CC BY-NC-SA 4.0. The license applies to both data and pretrained models.

## Contact
If you have any questions, please contact Yujian Liu ```<yujianl@umich.edu>``` or Xinliang Frederick Zhang ```<xlfzhang@umich.edu>``` or create a Github issue.

## Citation
Please cite our paper if you our **POLITICS** model and/or **BIGNEWS** dataset as well as their derivatives from this repo:
```
@inproceedings{liu-etal-2022-politics,
    title = "{POLITICS}: Pretraining with Same-story Article Comparison for Ideology Prediction and Stance Detection",
    author = "Liu, Yujian  and
      Zhang, Xinliang Frederick  and
      Wegsman, David  and
      Beauchamp, Nicholas  and
      Wang, Lu",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2022",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    pages = "1354--1374",
}
```

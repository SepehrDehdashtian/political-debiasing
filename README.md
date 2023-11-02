# Political Debiasing

```diff
- Content warning: This article contains examples of hateful text that might be disturbing, distressing, and/or offensive.
```

Mitigating Political Bias in Pre-trained Large Language Models

<img src="assets/cover.png" alt="Overview figure" width="400"/>

## Dataloader
The dataloader class is present [here](hal/datasets/HateSpeech/demographic_loader.py).

## Running the code
To run the code, you need to run the following command:

```bash
export CUDA_VISIBLE_DEVICES=0
python train.py --args exps/hatespeech/args_tmlr.ini --nepoch 10 --result-subdir hatespeech
```


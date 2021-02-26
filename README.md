# An Image Captioning codebase

This is a codebase for image captioning research.

It supports:
- Self critical training from [Self-critical Sequence Training for Image Captioning](https://arxiv.org/abs/1612.00563)
- Bottom up feature from [ref](https://arxiv.org/abs/1707.07998).
- Test time ensemble
- Multi-GPU training. (DistributedDataParallel is now supported with the help of pytorch-lightning, see [ADVANCED.md](ADVANCED.md) for details)
- Transformer captioning model.

A simple demo colab notebook is available [here](https://colab.research.google.com/github/ruotianluo/ImageCaptioning.pytorch/blob/colab/notebooks/captioning_demo.ipynb)

## Requirements
- Python 3
- PyTorch 1.3+ (along with torchvision)
- cider (already been added as a submodule)
- coco-caption (already been added as a submodule) (**Remember to follow initialization steps in coco-caption/README.md**)
- yacs
- lmdbdict

## Install

If you have difficulty running the training scripts in `tools`. You can try installing this repo as a python package:
```
python -m pip install -e .
```

## Pretrained models

Checkout [MODEL_ZOO.md](MODEL_ZOO.md).

If you want to do evaluation only, you can then follow [this section](#generate-image-captions) after downloading the pretrained models (and also the pretrained resnet101 or precomputed bottomup features, see [data/README.md](data/README.md)).

## Train your own network on COCO/Flickr30k

### Prepare data.

We now support both flickr30k and COCO. See details in [data/README.md](data/README.md). (Note: the later sections assume COCO dataset; it should be trivial to use flickr30k.)

### Start training

```bash
$ python tools/train.py --id fc --caption_model newfc --input_json data/cocotalk.json --input_fc_dir data/cocotalk_fc --input_att_dir data/cocotalk_att --input_label_h5 data/cocotalk_label.h5 --batch_size 10 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path log_fc --save_checkpoint_every 6000 --val_images_use 5000 --max_epochs 30
$ python tools/train.py --id fc --caption_model transformer --input_json data/cocotalk.json --input_fc_dir data/cocotalk_fc --input_att_dir data/cocotalk_att --input_label_h5 data/cocotalk_label.h5 --batch_size 10 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path log_fc --save_checkpoint_every 6000 --val_images_use 5000 --max_epochs 30
$ python tools/train.py --id fc --caption_model newfc --input_json data/f30ktalk.json --input_fc_dir data/flicktalk_fc --input_att_dir data/flicktalk_att --input_label_h5 data/f30ktalk_label.h5 --batch_size 10 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path log_fc_flick30 --save_checkpoint_every 6000 --val_images_use 5000 --max_epochs 30

```
python scripts/prepro_feats.py --input_json data/dataset_flickr30k.json --output_dir data/flicktalk --images_root $IMAGE_ROOT2
or 

```bash
$ python tools/train.py --cfg configs/fc.yml --id fc
```

The train script will dump checkpoints into the folder specified by `--checkpoint_path` (default = `log_$id/`). By default only save the best-performing checkpoint on validation and the latest checkpoint to save disk space. You can also set `--save_history_ckpt` to 1 to save every checkpoint.

To resume training, you can specify `--start_from` option to be the path saving `infos.pkl` and `model.pth` (usually you could just set `--start_from` and `--checkpoint_path` to be the same).

To checkout the training curve or validation curve, you can use tensorboard. The loss histories are automatically dumped into `--checkpoint_path`.

The current command use scheduled sampling, you can also set `--scheduled_sampling_start` to -1 to turn off scheduled sampling.

If you'd like to evaluate BLEU/METEOR/CIDEr scores during training in addition to validation cross entropy loss, use `--language_eval 1` option, but don't forget to pull the submodule `coco-caption`.

For all the arguments, you can specify them in a yaml file and use `--cfg` to use the configurations in that yaml file. The configurations in command line will overwrite cfg file if there are conflicts.  

For more options, see `opts.py`. 

<!-- **A few notes on training.** To give you an idea, with the default settings one epoch of MS COCO images is about 11000 iterations. After 1 epoch of training results in validation loss ~2.5 and CIDEr score of ~0.68. By iteration 60,000 CIDEr climbs up to about ~0.84 (validation loss at about 2.4 (under scheduled sampling)). -->

### Train using self critical

First you should preprocess the dataset and get the cache for calculating cider score:
```
$ python scripts/prepro_ngrams.py --input_json data/dataset_coco.json --dict_json data/cocotalk.json --output_pkl data/coco-train --split train
```

Then, copy the model from the pretrained model using cross entropy. (It's not mandatory to copy the model, just for back-up)
```
$ bash scripts/copy_model.sh fc fc_rl
```

Then
```bash
$ python tools/train.py --id fc_rl --caption_model newfc --input_json data/cocotalk.json --input_fc_dir data/cocotalk_fc --input_att_dir data/cocotalk_att --input_label_h5 data/cocotalk_label.h5 --batch_size 10 --learning_rate 5e-5 --start_from log_fc_rl --checkpoint_path log_fc_rl --save_checkpoint_every 6000 --language_eval 1 --val_images_use 5000 --self_critical_after 30 --cached_tokens coco-train-idxs --max_epoch 50 --train_sample_n 5
```

or 
```bash
$ python tools/train.py --cfg configs/fc_rl.yml --id fc_rl
```

You will see a huge boost on Cider score, : ).

**A few notes on training.** Starting self-critical training after 30 epochs, the CIDEr score goes up to 1.05 after 600k iterations (including the 30 epochs pertraining).

## Generate image captions

### Evaluate on raw images

**Note**: this doesn't work for models trained with bottomup feature.
Now place all your images of interest into a folder, e.g. `coco_test_dataset`, and run
the eval script:

```bash
$ python tools/eval.py --image_folder coco_test_dataset --num_images 10
```
**Attack**
Please note, the following attack modes set the default `--model` and `--infos`.
Next, we will provide several kinds of attack methods. If you want to modify the test model, please find the detail information from [model and infos section](#Model and infos)


- Targted Attack: Given one targeted image, generate adversarial examples that assign the same caption as this targeted image.
(Before run the following code, you should put one image into the folder of target_label.)
```
$ python tools/attack.py --image_folder coco_test_dataset --batch_size 1 --beam_size 3 --targeted
```
Next, if you want test other targeted attacks, `--targeted` should be set.
- Targeted_partical Attack: Given several words of a sent, complete the sent and generate the corresponding adversarial examples
```
$ python tools/attack.py --image_folder coco_test_dataset --batch_size 1 --beam_size 3 --targeted --targeted_partical
```
- Targeted_key_word Appear Attack: Given several words. These words will appear in the generated caption. However, we do not pre-set the available position.
```
$ python tools/attack.py --image_folder coco_test_dataset --batch_size 1 --beam_size 3 --targeted --targeted_key_word 'dog'
```
- Untargeted Attack: Generate new captions that different from the original caption.
```
$ python tools/attack.py --image_folder coco_test_dataset --batch_size 1 --beam_size 3
```
- Non-order Untargeted Attack: This mode is a more efficient untargeted attack, which eliminates the influence of the word position. 
```
$ python tools/attack.py --image_folder coco_test_dataset --batch_size 1 --beam_size 3 --non_order
```
- Key_word_disappear Attack: Given one word you wish to disappear (e.g. man), this word will not appear in the generated captions.
 (You can set any word that you wish to protect to --key_word_disappear)
```
$ python tools/attack.py --image_folder coco_test_dataset --batch_size 1 --beam_size 3 --key_word_disappear 'man'
```
- Key_word_disappear Attack 2: Given one word  you wish to disappear (e.g. man, You can set any word that you wish to protect to --key_word_disappear), the related words (e.g. man, men, women...) will not appear in the generated captions. Using --using_similarity_space
```
$ python tools/attack.py --image_folder coco_test_dataset --batch_size 1 --beam_size 3 --key_word_disappear 'man' --using_similarity_space
```
- Unseen_attack Untargeted Attack: We wish to maintain the meaning of the caption, but cover the special word. 
```
$ python tools/attack.py --image_folder coco_test_dataset --batch_size 1 --beam_size 3 --key_word_disappear 'man' --unseen_attack
```
- Ukd_mri Attack: untargeted keyword disappear + maintain rest information. We wish to maintain the meaning of the caption, but cover the special word. 
The difference between the Unseen_attack and Ukd_mri attack is:
    
    1. Ori: A car parked next to the house.
    2. keyword: car
    3. Unseen_attack: The house next to the parked vehicle.
    4. Ukd_mri: A vehicle parked next to the house.
```
$ python tools/attack.py --image_folder coco_test_dataset --batch_size 1 --beam_size 3 --key_word_disappear 'man' --ukd_mri
```
- Replace_word Attack: 
```
$ python tools/attack.py --image_folder coco_test_dataset --batch_size 1 --beam_size 3 --ukd_mri --key_word_disappear 'man' --replace_word 'elephant'
```
- Sup_word Attack
```
$python tools/attack.py --image_folder coco_test_dataset --batch_size 1 --beam_size 3 --sup_word 'black' --targeted  --model ./tools/model_show_and_tell.pth --infos_path ./tools/infos_show_and_tell.pkl
```
##Model and infos

The following models and infos will be given once our paper accepted
```
$ model_show_and_tell.pth  infos_show_and_tell.pkl
$ model_new_fc_self_cri.pth infos_new_fc_self_cri.pkl
$ model_new_atti_self_cri.pth  infos_new_atti_self_cri.pkl
$ model_transformer.pth  infos_transformer.pkl
```


This tells the `eval` script to run up to 10 images from the given folder. If you have a big GPU you can speed up the evaluation by increasing `batch_size`. Use `--num_images -1` to process all images. The eval script will create an `vis.json` file inside the `vis` folder, which can then be visualized with the provided HTML interface:

```bash
$ cd vis
$ python -m SimpleHTTPServer
```

Now visit `localhost:8000` in your browser and you should see your predicted captions.

### Evaluate on Karpathy's test split

```bash
$ python tools/eval.py --dump_images 0 --num_images 5000 --model model.pth --infos_path infos.pkl --language_eval 1 
```

The defualt split to evaluate is test. The default inference method is greedy decoding (`--sample_method greedy`), to sample from the posterior, set `--sample_method sample`.

**Beam Search**. Beam search can increase the performance of the search for greedy decoding sequence by ~5%. However, this is a little more expensive. To turn on the beam search, use `--beam_size N`, N should be greater than 1.

### Evaluate on COCO test set

```bash
$ python tools/eval.py --input_json cocotest.json --input_fc_dir data/cocotest_bu_fc --input_att_dir data/cocotest_bu_att --input_label_h5 none --num_images -1 --model model.pth --infos_path infos.pkl --language_eval 0
```

You can download the preprocessed file `cocotest.json`, `cocotest_bu_att` and `cocotest_bu_fc` from [link](https://drive.google.com/open?id=1eCdz62FAVCGogOuNhy87Nmlo5_I0sH2J).

## Miscellanea
**Using cpu**. The code is currently defaultly using gpu; there is even no option for switching. If someone highly needs a cpu model, please open an issue; I can potentially create a cpu checkpoint and modify the eval.py to run the model on cpu. However, there's no point using cpus to train the model.

**Train on other dataset**. It should be trivial to port if you can create a file like `dataset_coco.json` for your own dataset.

**Live demo**. Not supported now. Welcome pull request.

## For more advanced features:

Checkout [ADVANCED.md](ADVANCED.md).

## Reference

If you find this repo useful, please consider citing (no obligation at all):

```
@article{luo2018discriminability,
  title={Discriminability objective for training descriptive captions},
  author={Luo, Ruotian and Price, Brian and Cohen, Scott and Shakhnarovich, Gregory},
  journal={arXiv preprint arXiv:1803.04376},
  year={2018}
}
```

Of course, please cite the original paper of models you are using (You can find references in the model files).

## Acknowledgements

Thanks the original [neuraltalk2](https://github.com/karpathy/neuraltalk2) and awesome PyTorch team.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np
import sys
sys.path.append("./")
import time
import os
from six.moves import cPickle

import captioning.utils.opts as opts
import captioning.models as models
from captioning.data.dataloader import *
from captioning.data.dataloaderraw import *
import captioning.utils.eval_utils_for_attack as eval_utils_for_attack
import argparse
import captioning.utils.misc as utils
import captioning.modules.losses as losses
import torch

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model', type=str, default='./tools/model_new_atti_self_cri.pth',#-fc_nsc
                help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str,  default='resnet101',
                help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='./tools/infos_new_atti_self_cri.pkl',
                help='path to infos to evaluate')
parser.add_argument('--targeted',action='store_true',help='targeted attack or untargeted attack')
parser.add_argument('--targeted_sents_attack',action='store_true',help='targeted_sents_attack')
parser.add_argument('--non_order',action='store_true',help='non_order for untargeted attack')
parser.add_argument('--unseen_attack',action='store_true',help='unseen_attack for untargeted attack')
parser.add_argument('--ukd_mri',action='store_true',help='ukd_mri (untargeted keyword disappear + main rest information) for untargeted attack')
parser.add_argument('--using_similarity_space',action='store_true',help='using_similarity_space')
parser.add_argument('--sup_word',type=str, default='',help='given one word appear in the sents, and maintain the sent order')
parser.add_argument('--targeted_partical',action='store_true',help=' partical targeted attack')
parser.add_argument('--only_lang_eval', type=int, default=0,
                help='lang eval on saved results')
parser.add_argument('--imgs_num', type=int, default=1000,
                help='eval on how many images')
parser.add_argument('--iters', type=int, default=100,
                help='attack iterations')
parser.add_argument('--force', type=int, default=1,
                help='force to evaluate no matter if there are results available')
parser.add_argument('--device', type=str, default='cuda',
                help='cpu or cuda')
parser.add_argument('--targeted_path', type=str, default='./validation_dataset_coco/',
                help='path to infos to evaluate')
parser.add_argument('--similarity_set_path', type=str, default='data/similarity_vocab.npy',
                help='path to the directory containing the preprocessed similarity words')
parser.add_argument('--key_word_disappear', type=str, default='',
                help='key_word_disappear need to be pre-set')
parser.add_argument('--replace_word', type=str, default='',help='using this word to replace the key_word_disappear')
parser.add_argument('--targeted_key_word', type=str, default='',help='the word will be appear in sents')
parser.add_argument('--UCT_words', type=str, default='2',
                help='used in partical targeted attack, illustrate the un-constraint positions.')
opts.add_eval_options(parser)
opts.add_diversity_opts(parser)
opt = parser.parse_args()
f=open("targeted_OCWON.txt","w+")
sys.stdout=f
if opt.targeted_partical and opt.targeted_key_word:
    print("only one mode could be set to 1")
    sys.exit()
if opt.targeted:
    if opt.non_order:
        print("targeted for targeted attack, and non_order for untargeted attack, you are allowed to set targeted = True or non_order = True ")
        sys.exit()

if opt.unseen_attack:
    if not opt.key_word_disappear:
        print("unseen attack need to give the exact value to key_word_disappear  --key_word_disappear?")
        sys.exit()
# Load infos
with open(opt.infos_path, 'rb') as f:
    infos = utils.pickle_load(f)

# override and collect parameters
replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
ignore = ['start_from']

for k in vars(infos['opt']).keys():
    if k in replace:
        setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
        #print(getattr(infos['opt'], k, ''))
    elif k not in ignore:
        if not k in vars(opt):
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

vocab = infos['vocab'] # ix -> word mapping

pred_fn = os.path.join('eval_results/', '.saved_pred_'+ opt.id + '_' + opt.split + '.pth')
result_fn = os.path.join('eval_results/', opt.id + '_' + opt.split + '.json')

if opt.only_lang_eval == 1 or (not opt.force and os.path.isfile(pred_fn)): 
    # if results existed, then skip, unless force is on
    if not opt.force:
        try:
            if os.path.isfile(result_fn):
                print(result_fn)
                json.load(open(result_fn, 'r'))
                print('already evaluated')
                os._exit(0)
        except:
            pass

    predictions, n_predictions = torch.load(pred_fn)
    lang_stats = eval_utils.language_eval(opt.input_json, predictions, n_predictions, vars(opt), opt.split)
    print(lang_stats)
    os._exit(0)

# At this point only_lang_eval if 0
if not opt.force:
    # Check out if 
    try:
        # if no pred exists, then continue
        tmp = torch.load(pred_fn)
        # if language_eval == 1, and no pred exists, then continue
        if opt.language_eval == 1:
            json.load(open(result_fn, 'r'))
        print('Result is already there')
        os._exit(0)
    except:
        pass

# Setup the model
opt.vocab = vocab
#print("caption_model:",opt.caption_model)
model = models.setup(opt)
del opt.vocab
model.load_state_dict(torch.load(opt.model, map_location='cpu'))
model.to(opt.device)
model.eval()


#a = torch.tensor([['2','38','64']])
#seq,_ = utils.decode_sequence_with_len(model.vocab, a)
#print(seq)
crit = losses.LanguageModelCriterion()

# Create the Data Loader instance
if len(opt.image_folder) == 0:
    loader = DataLoader(opt)
else:
    loader = DataLoaderRaw({'folder_path': opt.image_folder, 
                            'coco_json': opt.coco_json,
                            'batch_size': opt.batch_size,
                            'cnn_model': opt.cnn_model})
# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.dataset.ix_to_word = infos['vocab']
# Set sample options
opt.dataset = opt.input_json

eval_utils_for_attack.eval_split(model, crit, loader,opt)
print(opt.model)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F

import numpy as np
import json

import os
import sys
from . import misc as utils
import skimage.io
from captioning.utils.l2_attack import CarliniWagnerL2
import codecs

import spacy
import imageio
from readability_score.calculators.colemanliau import *
nlp = spacy.load('en_core_web_lg')

# load coco-caption if available
try:
    sys.path.append("coco-caption")
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
except:
    print('Warning: coco-caption not available')

bad_endings = ['a', 'an', 'the', 'in', 'for', 'at', 'of', 'with', 'before', 'after', 'on', 'upon', 'near', 'to', 'is',
               'are', 'am']
bad_endings += ['the']

sub_init = torch.Tensor([0.485, 0.456, 0.406])
sub_init = sub_init.reshape([3, 1, 1])
sub_div = torch.Tensor([0.229, 0.224, 0.225])
sub_div = sub_div.reshape([3, 1, 1])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def count_bad(sen):
    sen = sen.split(' ')
    if sen[-1] in bad_endings:
        return 1
    else:
        return 0


def getCOCO(dataset):
    if 'coco' in dataset:
        annFile = 'coco-caption/annotations/captions_val2014.json'
    elif 'flickr30k' in dataset or 'f30k' in dataset:
        annFile = 'data/f30k_captions4eval.json'
    return COCO(annFile)


def language_eval(dataset, preds, preds_n, eval_kwargs, split):
    model_id = eval_kwargs['id']
    eval_oracle = eval_kwargs.get('eval_oracle', 0)

    # create output dictionary
    out = {}

    if len(preds_n) > 0:
        # vocab size and novel sentences
        if 'coco' in dataset:
            dataset_file = 'data/dataset_coco.json'
        elif 'flickr30k' in dataset or 'f30k' in dataset:
            dataset_file = 'data/dataset_flickr30k.json'
        training_sentences = set([' '.join(__['tokens']) for _ in json.load(open(dataset_file))['images'] if
                                  not _['split'] in ['val', 'test'] for __ in _['sentences']])
        generated_sentences = set([_['caption'] for _ in preds_n])
        novels = generated_sentences - training_sentences
        out['novel_sentences'] = float(len(novels)) / len(preds_n)
        tmp = [_.split() for _ in generated_sentences]
        words = []
        for _ in tmp:
            words += _
        out['vocab_size'] = len(set(words))

    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    cache_path = os.path.join('eval_results/', '.cache_' + model_id + '_' + split + '.json')

    coco = getCOCO(dataset)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set
    preds_filt = [p for p in preds if p['image_id'] in valids]
    mean_perplexity = sum([_['perplexity'] for _ in preds_filt]) / len(preds_filt)
    mean_entropy = sum([_['entropy'] for _ in preds_filt]) / len(preds_filt)
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w'))  # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    for metric, score in cocoEval.eval.items():
        out[metric] = score
    # Add mean perplexity
    out['perplexity'] = mean_perplexity
    out['entropy'] = mean_entropy

    imgToEval = cocoEval.imgToEval
    for k in list(imgToEval.values())[0]['SPICE'].keys():
        if k != 'All':
            out['SPICE_' + k] = np.array([v['SPICE'][k]['f'] for v in imgToEval.values()])
            out['SPICE_' + k] = (out['SPICE_' + k][out['SPICE_' + k] == out['SPICE_' + k]]).mean()
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption

    if len(preds_n) > 0:
        from . import eval_multi
        cache_path_n = os.path.join('eval_results/', '.cache_' + model_id + '_' + split + '_n.json')
        allspice = eval_multi.eval_allspice(dataset, preds_n, model_id, split)
        out.update(allspice['overall'])
        div_stats = eval_multi.eval_div_stats(dataset, preds_n, model_id, split)
        out.update(div_stats['overall'])
        if eval_oracle:
            oracle = eval_multi.eval_oracle(dataset, preds_n, model_id, split)
            out.update(oracle['overall'])
        else:
            oracle = None
        self_cider = eval_multi.eval_self_cider(dataset, preds_n, model_id, split)
        out.update(self_cider['overall'])
        with open(cache_path_n, 'w') as outfile:
            json.dump({'allspice': allspice, 'div_stats': div_stats, 'oracle': oracle, 'self_cider': self_cider},
                      outfile)

    out['bad_count_rate'] = sum([count_bad(_['caption']) for _ in preds_filt]) / float(len(preds_filt))
    outfile_path = os.path.join('eval_results/', model_id + '_' + split + '.json')
    with open(outfile_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out


def onehot_2_label(target_seq,opt):
    if len(target_seq.shape) > 1:
        target_seq = torch.squeeze(target_seq, 0).unsqueeze(1)
    else:
        target_seq = target_seq.unsqueeze(1)
    seq_len = len(target_seq)
    # target_seq = torch.LongTensor([[7961],[5001],[9084],[9196],[9084]])
    target_cap = torch.zeros(seq_len, opt.vocab_size+1).scatter_(1, target_seq.cpu(), 1)
    labels = torch.Tensor(target_cap).to(device)
    return labels


def eval_split(model, crit, loader, opt):
    eval_kwargs = vars(opt)
    verbose_beam = eval_kwargs.get('verbose_beam', 0)
    split = eval_kwargs.get('split', 'val')
    beam_size = eval_kwargs.get('beam_size', 1)
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
    os.environ["REMOVE_BAD_ENDINGS"] = str(
        remove_bad_endings)  # Use this nasty way to make other code clean since it's a global configuration
    device = eval_kwargs.get('device', 'cuda')
    attack = CarliniWagnerL2((0.0, 1.0), 1000, learning_rate=0.01, search_steps=1, max_iterations=opt.iters,
                             initial_const=100, quantize=False, device=device)
    attack.eval_kwargs = eval_kwargs
    attack.my_resnet = loader.my_resnet
    # for name, parms in attack.my_resnet.named_parameters():
    #    print('-->name:', name, '-->grad_requirs:',parms.requires_grad,' -->grad_value:',parms.grad)
    # Make sure in the evaluation mode
    model.eval()
    f = open("vocab.txt", 'w+')
    for i in range(1, len(model.vocab) + 1):
        f.write(model.vocab[str(i)] + '\n')
    f.close()
    loader.reset_iterator(split)
    sum_suc = 0
    sum_pre = 0
    sum_recall = 0
    sum_l2_noise = 0
    sum_Score = 0
    sum_TRR = 0
    sum_TPR = 0
    if opt.targeted_partical:
        sum_PWRR = 0
    if opt.targeted_key_word or opt.key_word_disappear:
        sum_KAR = 0
        sum_keyword = 0
    if opt.sup_word:
        sum_KAR = 0
        sum_TRR = 0
        sum_TPR = 0
    print("---------------------------test attack--------------------------------")
    index_cap = 0
    Key = list(model.vocab.keys())
    Value = list(model.vocab.values())
    for img_index in range(opt.imgs_num):
        imgs = loader.get_batch_attack()
        images = imgs.to(device)
        if opt.targeted:
            img_ind = img_index+1#np.random.randint(3000, 4000)
            # -- targeted_path ./validation_dataset_coco/
            targeted_img_name = os.path.join(opt.targeted_path, str(img_ind) + ".png")
            img = skimage.io.imread(targeted_img_name)
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
                img = np.concatenate((img, img, img), axis=2)
            img = img[:, :, :3].astype('float32') / 255.0
            img = torch.from_numpy(img.transpose([2, 0, 1]))  # .cuda()
            img = (img - sub_init) / sub_div  # preprocess(img)
            img = img.cuda()
            target_seq, target_logits = attack._eval(model, img)
            target_sents, real_len = utils.decode_sequence_with_len(model.vocab, target_seq)
            
            index_cap += 1
            while True:
              try:
                caption_sup = json.load(open("./data/dataset_coco.json"))['images'][index_cap]['sentences'][0]['tokens']
                target_sents = json.load(open("./data/dataset_coco.json"))['images'][index_cap]['sentences'][0]['raw']
                #print(caption_sup)
                target_seq = []
                real_len = len(caption_sup)
                for i in range(real_len):
                    target_seq.append(int(Key[Value.index(caption_sup[i])]))
                if real_len <20:
                    break
              except ValueError:
                index_cap += 1
                print("Value Error")
            target_seq = torch.tensor(target_seq).unsqueeze(0).cuda()
            labels = onehot_2_label(target_seq,opt)
            seq_len = opt.max_length
            print("targeted:", target_sents)
            valid_tar_seq = target_seq[0][:real_len]
            if opt.targeted_partical:
                word_positions = opt.UCT_words.split()
                word_position = int(word_positions[0])
                partical_word = target_sents[0].split()[word_position]
                partical_word = nlp(str(partical_word))

                word_att = list(partical_word.sents)[0][0].tag_
                filename = "attribute/" + str(word_att) + '.txt'
                document = codecs.open(filename, 'r', encoding='utf-8').read()
                document = nlp(document)
                l_sent = list(document.sents)
                potential_set = []
                for i in range(len(l_sent)):
                    for word in l_sent[i]:
                        if not word.text == "\n":
                            potential_set.append(word.text)
            else:
                potential_set = []
                seq_len = real_len
        ori_seq, ori_logits = attack._eval(model, images)
        if not opt.targeted:
            potential_set = []
            labels = onehot_2_label(ori_seq,opt)
            sents, seq_len = utils.decode_sequence_with_len(model.vocab, ori_seq)
        else:
            sents, _= utils.decode_sequence_with_len(model.vocab, ori_seq)

        ori_sents = sents[0]
        print("ori_sample:", sents)
        #if opt.targeted_key_word in ori_sents or opt.key_word_disappear in ori_sents:
        #    sum_keyword += 1
        #else:
        #    continue
        #sup_word_list = ['people','cat','cow','white','orange','umbrella','pool','hat','on','in','beautiful','colorful']
        #index = np.random.randint(len(sup_word_list))
        #opt.sup_word = sup_word_list[index]

        seq, logits ,l2_noise,noise,adv_print= attack.attack(model, images, labels, potential_set, seq_len, opt)
        #imageio.imsave('./store/unkeydis_noise/'+str(img_index)+'.png',noise.cpu().transpose(0,1).transpose(1,2))
        #imageio.imsave('./store/unkeydis_adv/'+str(img_index)+'.png',adv_print.cpu().transpose(0,1).transpose(1,2))
        sents, adlen = utils.decode_sequence_with_len(model.vocab, seq)
        valid_ad_seq = seq[0][:adlen]
        if opt.targeted:
            if opt.targeted_partical:
                suc,pre,recall,PWRR = eval_for_tpa(valid_tar_seq,valid_ad_seq,opt)
                sum_PWRR += PWRR
            else:
                suc,pre,recall = eval_for_tsa(valid_tar_seq,valid_ad_seq)
        else:
            suc,pre,recall = eval_for_untarattack(ori_seq[0][:seq_len],valid_ad_seq,opt)
        
        sum_suc += suc
        sum_pre += pre
        sum_l2_noise += l2_noise
        sum_recall += recall
        print("adv_sample", sents)
        if opt.targeted_partical:
            print("index: %.1f, sum_suc: %.4f, sum_pre: %.4f, sum_recall: %.4f,sum_PWRR: %.4f, l2_noise: %.4f"%(img_index+1,sum_suc,sum_pre,sum_recall,sum_PWRR,sum_l2_noise))
        elif opt.targeted_key_word or opt.key_word_disappear:
            score = ColemanLiau(sents[0], locale='nl_NL').min_age
            if detect(sents[0]):
                score = 0
            sum_Score += score
            if opt.targeted_key_word in sents[0] or opt.key_word_disappear in sents[0]:
                sum_KAR += 1
            print("index: %.1f, sum_keyword: %.4f, sum_pre: %.4f, sum_recall: %.4f,sum_KAR: %.4f,sum_Score: %.4f, l2_noise: %.4f"%(sum_keyword,sum_keyword,sum_pre,sum_recall,sum_KAR,sum_Score/(sum_keyword+1),sum_l2_noise/(sum_keyword+1)))
        elif opt.sup_word:
            TPR,TRR = eval_for_sup(target_sents[0],sents[0])
            sum_TPR += TPR
            sum_TRR += TRR
            if opt.sup_word in sents[0]:
                sum_KAR += 1
            print("index: %.1f, sum_suc: %.4f, sum_pre: %.4f, sum_recall: %.4f,sum_TPR: %.4f, sum_TRR: %.4f,sum_KAR: %.4f, l2_noise: %.4f"%(img_index+1,sum_suc,sum_pre,sum_recall,sum_TPR,sum_TRR,sum_KAR,sum_l2_noise))
        else:
            fk = ColemanLiau(sents[0], locale='nl_NL')
            score = fk.min_age
            sum_Score += score
            TPR,TRR = eval_for_sup(ori_sents,sents[0])
            sum_TPR += TPR
            sum_TRR += TRR
            print("index: %.1f, sum_suc: %.4f, sum_pre: %.4f, sum_recall: %.4f,sum_TPR: %.4f, sum_TRR: %.4f,sum_Score: %.4f, l2_noise: %.4f"%(img_index+1,sum_suc,sum_pre,sum_recall,sum_TPR,sum_TRR,sum_Score,sum_l2_noise))
            #if detect(sents[0]):
            #    score = 0
            #sum_Score += score
            #print("index: %.1f, sum_suc: %.4f, sum_pre: %.4f,sum_Score: %.4f, sum_recall: %.4f,l2_noise: %.4f"%(img_index+1,sum_suc,sum_pre,sum_Score,sum_recall,sum_l2_noise))
        print("-----------------------------------------------------------------------")
# evaluation metrics for targted sents attack
def detect(sents):
    words = sents.split()
    index = 0
    for i in range(len(words)-1):
        if words[i] == words[i+1]:
            index += 1
    if index > len(words)/2:
        return True
    else:
        return False
def eval_for_sup(ori_sent,adv_sent):
    sent_list = ori_sent.split()
    adv_list = adv_sent.split()
    TRR = 0
    for i in range(len(sent_list)-1):
        if sent_list[i]+' '+sent_list[i+1] in adv_sent:
            TRR += 1
    if sent_list[0]+' '+sent_list[-1] == adv_list[0]+' '+adv_list[-1]:
        TRR += 1
    TPR = TRR / len(adv_list)
    TRR = TRR / len(sent_list)
    return TPR,TRR


def eval_for_tpa(tar,generate,opt):
    word_positions = opt.UCT_words.split()
    word_position = int(word_positions[0])
    PWRR = 0
    if tar[word_position] == generate[word_position]:
        PWRR = 1
    tar[word_position] = 0
    generate[word_position] = 0
    if len(tar) == len(generate):
        if (tar == generate).all():
            suc = 1.
            pre = 1.
            recall = 1.
        else:
            suc = 0.
            pre = 0.
            for i in range(len(tar)):
                if tar[i] in generate:
                    pre += 1
            recall = (pre -1) / (len(tar)-1)
            pre= (pre-1)/(len(generate)-1)
    else:
        suc = 0.
        pre = 0.
        for i in range(len(tar)):
            if tar[i] in generate:
                pre += 1
        recall = (pre -1) / (len(tar)-1)
        pre= (pre-1)/(len(generate)-1)
    return suc,pre, recall,PWRR

def eval_for_tsa(tar,generate):
    if len(tar) == len(generate):
        if (tar == generate).all():
            suc = 1.
            pre = 1.
            recall = 1.
        else:
            suc = 0.
            pre = 0.
            for i in range(len(tar)):
                if tar[i] in generate:
                    pre += 1
            recall = pre / len(tar)
            pre /= len(generate)
    else:
        suc = 0.
        pre = 0.
        for i in range(len(tar)):
            if tar[i] in generate:
                pre += 1
        recall = pre / len(tar)
        pre /= len(generate)
    return suc,pre, recall

def eval_for_untarattack(tar,generate,opt):
    if '30k' in opt.image_folder:
        meanliss_filter = [28, 70, 4744, 322, 48, 727, 830, 88, 914, 879, 500, 3458, 6428, 443, 1945, 2807, 532, 3190, 84, 121, 5451, 19, 102, 334, 8, 2719, 4799, 1137, 215, 697, 3396, 888, 4207, 459, 4297, 1883, 325, 1108, 170, 392, 1399, 3083, 332, 3152, 5784, 2647, 1104, 6613, 680, 224, 1590, 1732, 2653, 6183, 5543, 2666, 1568, 5743, 3453, 185, 436, 56, 1734, 46, 4822, 6108, 3441, 1721, 2144, 3874, 3076, 1907, 14, 5235, 305, 76, 66, 262, 2251, 444, 2376, 1277, 5369, 21, 3061, 184, 217, 51, 337, 49, 5689, 95, 275, 644, 13, 212, 6861, 20, 174, 839, 2478, 1990, 1588, 5379, 3506, 4910, 6173, 613, 762, 1109, 1248, 257, 1662, 1276, 3182, 2212, 2215, 3835, 3100, 4481, 3282, 1417, 670, 15, 949, 891, 750, 804, 60, 194, 79, 233, 688, 6225, 3818, 1, 848, 2081, 47, 1023, 4579, 950, 6358, 6417, 1159, 6519, 1165, 11, 838, 409, 3414, 4, 2511, 177, 2362, 4721]
    if 'coco' in opt.image_folder:
        meanliss_filter = [1, 60, 340, 452, 577, 335, 2364, 834, 159, 236, 1254, 4084, 2046, 3322, 809, 5976, 967, 1715, 28, 486, 2859, 179, 173, 127, 79, 6552, 1336, 2190, 477, 1217, 294, 381, 1504, 661, 3301, 3520, 241, 231, 114, 824, 2643, 4304, 1178, 2120, 529, 3208, 1433, 4998, 1480, 599, 2513, 3099, 5351, 5160, 4504, 2083, 2921, 3615, 2838, 2078, 307, 995, 8093, 273, 2745, 3419, 5200, 1150, 2298, 8257, 7636, 1228, 32, 2293, 367, 245, 35, 225, 1751, 565, 215, 1389, 7046, 119, 4372, 556, 517, 17, 408, 6, 4415, 90, 494, 931, 139, 803, 6991, 591, 320, 1085, 730, 3441, 3939, 3346, 2028, 1477, 718, 1637, 3201, 128, 1937, 156, 1944, 2655, 1262, 1470, 1624, 4017, 5662, 3818, 2566, 1409, 50, 14, 1567, 182, 4637, 3716, 417, 553, 23, 445, 1036, 3093, 6148, 167, 472, 1091, 86, 1768, 308, 184, 4870, 5389, 576, 6402, 2528, 70, 538, 69, 2691, 3, 1927, 549, 4530, 4203]


    tar_filter = []
    generate_filter = []
    for i in range(len(tar)):
        if tar[i] not in meanliss_filter:
            tar_filter.append(tar[i])
    for i in range(len(generate)):
        if generate[i] not in meanliss_filter:
            generate_filter.append(generate[i])
    if len(tar) == len(generate):
        if (tar == generate).all():
            suc = 1.
            pre = 1.
            recall = 1.
        elif len(tar_filter) == 0 or len(generate_filter) == 0:
            suc = 0.
            pre = 0.
            recall = 0.
        else:
            suc = 0.
            pre = 0.
            for i in range(len(tar_filter)):
                if tar_filter[i] in generate_filter:
                    pre += 1
            recall = pre / len(tar_filter)
            pre /= len(generate_filter)
    else:
        if len(tar_filter) == 0 or len(generate_filter) == 0:
            suc = 0.
            pre = 0.
            recall = 0.
        else:
            suc = 0.
            pre = 0.
            for i in range(len(tar_filter)):
                if tar_filter[i] in generate_filter:
                    pre += 1
            recall = pre / len(tar_filter)
            pre /= len(generate_filter)
    return suc,pre, recall


# Only run when sample_n > 0
def eval_split_n(model, n_predictions, input_data, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    beam_size = eval_kwargs.get('beam_size', 1)
    sample_n = eval_kwargs.get('sample_n', 1)
    sample_n_method = eval_kwargs.get('sample_n_method', 'sample')

    fc_feats, att_feats, att_masks, data = input_data

    tmp_eval_kwargs = eval_kwargs.copy()
    if sample_n_method == 'bs':
        # case 1 sample_n == beam size
        tmp_eval_kwargs.update({'sample_n': 1, 'beam_size': sample_n, 'group_size': 1})  # randomness from softmax
        with torch.no_grad():
            model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
        for k in range(fc_feats.shape[0]):
            _sents = utils.decode_sequence(model.vocab,
                                           torch.stack([model.done_beams[k][_]['seq'] for _ in range(sample_n)]))
            for sent in _sents:
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                n_predictions.append(entry)
    # case 2 sample / gumbel / topk sampling/ nucleus sampling
    elif sample_n_method == 'sample' or \
            sample_n_method == 'gumbel' or \
            sample_n_method.startswith('top'):
        tmp_eval_kwargs.update(
            {'sample_n': sample_n, 'sample_method': sample_n_method, 'beam_size': 1})  # randomness from sample
        with torch.no_grad():
            _seq, _sampleLogprobs = model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
        _sents = utils.decode_sequence(model.vocab, _seq)
        _perplexity = - _sampleLogprobs.gather(2, _seq.unsqueeze(2)).squeeze(2).sum(1) / (
                (_seq > 0).to(_sampleLogprobs).sum(1) + 1)
        for k, sent in enumerate(_sents):
            entry = {'image_id': data['infos'][k // sample_n]['id'], 'caption': sent,
                     'perplexity': _perplexity[k].item()}
            n_predictions.append(entry)
    elif sample_n_method == 'dbs':
        # Use diverse beam search
        tmp_eval_kwargs.update({'beam_size': sample_n * beam_size, 'group_size': sample_n})  # randomness from softmax
        with torch.no_grad():
            model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
        for k in range(loader.batch_size):
            _sents = utils.decode_sequence(model.vocab, torch.stack(
                [model.done_beams[k][_]['seq'] for _ in range(0, sample_n * beam_size, beam_size)]))
            for sent in _sents:
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                n_predictions.append(entry)
    else:
        tmp_eval_kwargs.update(
            {'sample_method': sample_n_method[1:], 'group_size': sample_n, 'beam_size': 1})  # randomness from softmax
        with torch.no_grad():
            _seq, _sampleLogprobs = model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
        _sents = utils.decode_sequence(model.vocab, _seq)
        for k, sent in enumerate(_sents):
            entry = {'image_id': data['infos'][k // sample_n]['id'], 'caption': sent}
            n_predictions.append(entry)
    if verbose:
        for entry in sorted(n_predictions[-fc_feats.shape[0] * sample_n:], key=lambda x: x['image_id']):
            print('image %s: %s' % (entry['image_id'], entry['caption']))
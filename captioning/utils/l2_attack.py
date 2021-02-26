from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from . import misc as utils
from . import eval_utils_for_attack as EUA
import sys


Similarity_space = [u'a',u'an',u'aboard', u'about', u'above', u'across', u'after', u'against', u'all', u'along', u'almost', u'already', u'always', u'am', u'amid', u'amidst', u'among', u'amongst', u'and', u'another', u'any', u'are', u'around', u'as', u'at', u'await', u'because', u'before', u'behind', u'below', u'beneath', u'beside', u'besides', u'between', u'better', u'beyond', u'both', u'but', u'by', u'can' , u'could',u'completely', u'clearly', u'curly', u'currently', u'despite', u'directly', u'dude', u'during', u'each', u'eight', u'either', u'eleven', u'every', u'except', u'early', u'everywhere', u'enough', u'else', u'five', u'for', u'four', u'fourteen', u'from', u'fully', u'fairly', u'hello', u'here',u'how', u'hi', u'highly', u'if', u'in', u'include', u'inside', u'into', u'is', u'it', u'itself', u'like', u'may', u'must', u'might', u'near', u'nine', u'no', u'not', u'of', u'off', u'on', u'once', u'one', u'onto', u'or', u'out', u'outdoors', u'outfielder', u'outside', u'over', u'partially',u'possibly', u'probably', u'quickly',u'quite', u'really', u'rather',u'retro',u'recently', u'seven', u'she', u'six', u'some', u'still', u'so', u'slightly', u'somewhere', u'somewhat', u'should', u'seemingly', u'sure', u'ten', u'than', u'that', u'the', u'these', u'this', u'those', u'though', u'three', u'through', u'to', u'toward', u'towards', u'twelve', u'twenty', u'two', u'under', u'underneath', u'up', u'upon', u'us',  u'very', u'via', u'want', u'well', u'why', u'what', u'while', u'where', u'who', u'whom', u'with', u'within', u'without',u'would', u'yet']

class CarliniWagnerL2:
    """
    Carlini's attack (C&W): https://arxiv.org/abs/1608.04644
    Based on https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks_tf.py
    Parameters
    ----------
    image_constraints : tuple
        Bounds of the images.
    num_classes : int
        Number of classes of the model to attack.
    confidence : float, optional
        Confidence of the attack for Carlini's loss, in term of distance between logits.
    learning_rate : float
        Learning rate for the optimization.
    search_steps : int
        Number of search steps to find the best scale constant for Carlini's loss.
    max_iterations : int
        Maximum number of iterations during a single search step.
    initial_const : float
        Initial constant of the attack.
    quantize : bool, optional
        If True, the returned adversarials will have possible values (1/255, 2/255, etc.).
    device : torch.device, optional
        Device to use for the attack.
    callback : object, optional
        Callback to display losses.
    """

    def __init__(self,
                 image_constraints: Tuple[float, float],
                 num_classes: int,
                 confidence: float = 0,
                 learning_rate: float = 0.01,
                 search_steps: int = 9,
                 max_iterations: int = 10000,
                 abort_early: bool = True,
                 initial_const: float = 0.001,
                 quantize: bool = False,
                 device: torch.device = torch.device('cpu'),
                 callback: Optional = None) -> None:

        self.confidence = confidence
        self.learning_rate = learning_rate

        self.binary_search_steps = search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.num_classes = num_classes

        self.repeat = self.binary_search_steps >= 10

        self.boxmin = image_constraints[0]
        self.boxmax = image_constraints[1]
        self.boxmul = (self.boxmax - self.boxmin) / 2
        self.boxplus = (self.boxmin + self.boxmax) / 2
        self.quantize = quantize
        sub_init = torch.Tensor([0.485, 0.456, 0.406])
        self.sub_init = sub_init.reshape([3, 1, 1])
        sub_div = torch.Tensor([0.229, 0.224, 0.225])
        self.sub_div = sub_div.reshape([3, 1, 1])
        self.device = device
        self.callback = callback
        self.log_interval = 10

    @staticmethod
    def _arctanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x *= (1. - eps)
        return (torch.log((1 + x) / (1 - x))) * 0.5

    def _step(self, model: nn.Module, optimizer: optim.Optimizer, inputs: torch.Tensor, tinputs: torch.Tensor,
              modifier: torch.Tensor, labels: torch.Tensor, labels_infhot: torch.Tensor, targeted: bool,
              const: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size = inputs.shape[0]

        adv_input = torch.tanh(tinputs + modifier) * self.boxmul + self.boxplus
        l2 = (adv_input - inputs).view(batch_size, -1).pow(2).sum(1)

        data = self.get_feature(adv_input)
        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
        tmp = [_.to('cuda') if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, labels, masks, att_masks = tmp
        # forward the model to also get generated samples for each image

        tmp_eval_kwargs = self.eval_kwargs.copy()
        tmp_eval_kwargs.update({'sample_n': 1})
        seq, logits = model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
        seq = seq.data
        logit_dists = 0
        if len(self.target_cap.shape) == 1:
            for i in range(self.target_cap.shape[0]):
                logits_i = logits.squeeze(0)[i]

                logit_dists += logits_i[int(self.target_cap[i])]
        elif len(self.target_cap.shape) == 2:
            for i in range(self.target_cap.shape[1]):
                logit_dists += logits[i][self.target_cap[0, i]]
        else:
            print("Wrong Target Label")

        logit_loss = (const * logit_dists).sum()
        l2_loss = l2.sum()
        loss = logit_loss # + l2_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return adv_input.detach(), logits.detach(), l2.detach(), logit_dists.detach(), loss.detach(), modifier

    def get_NN(self, model, path='NNPS.txt'):
        with open(path, 'r') as f:
            NN_set = f.readlines()
        # print(len(list1))
        for i in range(0, len(NN_set)):
            NN_set[i] = NN_set[i].strip('\n')
        key_word_list = []
        for i in range(len(NN_set)):
            if int(self.Key[self.Value.index(NN_set[i])]) == 9487:
                pass
            else:
                key_word_list.append(int(self.Key[self.Value.index(NN_set[i])]))
        return key_word_list, NN_set

    def found_sub_label(self, model, label, potential_set, inputs, word_position,opt):
        vocab_set = []
        data = self.get_feature(inputs)

        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
        tmp = [_.to('cuda') if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, _, masks, att_masks = tmp
        tmp_eval_kwargs = self.eval_kwargs.copy()
        tmp_eval_kwargs.update({'sample_n': 3})
        for i in range(len(potential_set)):  # opt.vocab_size
            # perform the attac


            # forward the model to also get generated samples for each image
            if opt.beam_size == 3:
                value = torch.tensor([potential_set[i],potential_set[i],potential_set[i]]).cuda()
            else:
                value = torch.tensor(potential_set[i]).cuda()
            seq, logits = model(fc_feats, att_feats, att_masks, True, word_position, value, opt=tmp_eval_kwargs,
                                mode='sample')
            logit_dists = (label[:] * logits.squeeze(0)[:len(label)]).sum()

            vocab_set.append(logit_dists.clone().detach())
        rep_max_index = np.argmax(vocab_set)

        return potential_set[rep_max_index]
    def split_joint(self,joint_label):
        seq_len = joint_label.shape[0]
        sup_labels = torch.tensor([]).cuda()
        for i in range(seq_len-1):
            sup_label = torch.cat((joint_label[i].unsqueeze(0),joint_label[i+1].unsqueeze(0)),0)
            sup_labels = torch.cat((sup_label.unsqueeze(0),sup_labels),0)
        sup_label = torch.cat((joint_label[0].unsqueeze(0),joint_label[-1].unsqueeze(0)),0)
        sup_labels = torch.cat((sup_label.unsqueeze(0),sup_labels),0)
        return sup_labels
    def process_single_label(self,opt,word):
        key_word_list = []
        key_word_list.append(int(self.Key[self.Value.index(word)]))
        key_word_list = torch.tensor(key_word_list)
        label = EUA.onehot_2_label(key_word_list,opt)
        return label
    def define_label(self, tar_labels, att_set, inputs, model, seq_len, opt, search=False):
        Key = model.vocab.keys()
        self.Key = list(Key)
        Value = model.vocab.values()
        self.Value = list(Value)
        potential_set = []
        for i in range(len(att_set)):
            try:
                potential_set.append(int(self.Key[self.Value.index(att_set[i])]))
            except ValueError:
                pass
        # attack mode 1 : targeted attack
        if opt.targeted:
            label = tar_labels[:seq_len].clone().detach()
            # attack mode 6 : targeted partical attack
            if opt.targeted_partical:
                if not search:
                    word_positions = opt.UCT_words.split()
                    word_position = int(word_positions[0])
                    init_zeros = torch.zeros_like(tar_labels[0])
                    label[word_position] = init_zeros
                else:
                    print("setting search == True")
                    word_positions = opt.UCT_words.split()
                    word_position = int(word_positions[0])
                    sub_label = self.found_sub_label(model, tar_labels, potential_set, inputs, word_position,opt)
                    init_zeros = torch.zeros_like(tar_labels[0])
                    init_zeros[sub_label] = 1
                    label[word_position] = init_zeros
            # attack mode 4: targeted key word appear attack
            if opt.targeted_key_word:
                label = self.process_single_label(opt,opt.targeted_key_word)
                label = torch.sum(label,0)
            label = -label
            if opt.sup_word:
                joint_label = -tar_labels[:seq_len]
                label = self.split_joint(joint_label)
   
        else:
            # attack mode 7: Non-targeted attack
            label = tar_labels[1:seq_len]
            # attack mode 8: Non-order-untargeted attackfdd
            if opt.non_order:
                label = torch.sum(tar_labels[1:seq_len], 0).unsqueeze(0)
                value_set = []
                for i in range(len(Similarity_space)):
                    value_set.append(int(self.Key[self.Value.index(Similarity_space[i])]))
                    # not modify the first word "a", it must appear in the initial sent
                #print(value_set)
                value_set = torch.tensor(value_set)
                value_set = EUA.onehot_2_label(value_set,opt)
                mf_filter = torch.sum(value_set,0)
                label = (1-mf_filter)*label[0].unsqueeze(0)
            if opt.key_word_disappear:
                key_words_disappear = opt.key_word_disappear.split()
                key_word_list = []
                # attack mode 10: --using similarity space
                if opt.using_similarity_space:
                    similarity_set = None
                    if similarity_set is None:
                        similarity_set = np.load(opt.similarity_set_path, allow_pickle=True, encoding="latin1").item()
                    for key_word_disappear in (key_words_disappear):
                        key_word_disappear = str(key_word_disappear)
                        try:
                            pickcle = similarity_set[key_word_disappear]
                        except KeyError:
                            print("similarity_set does not contain the key of %s" % key_word_disappear)
                        else:
                            for i in range(len(pickcle)):
                                key_word_list.append(int(self.Key[self.Value.index(pickcle[i][0])]))
                # attack mode 9:  key word disappear attack
                else:
                    key_word_list.append(int(self.Key[self.Value.index(opt.key_word_disappear)]))
                key_word_list = torch.tensor(key_word_list)
                label = EUA.onehot_2_label(key_word_list,opt)
                key_word_disappear_label = label
                # print(torch.argmax(key_word_disappear_label,1))
                label = torch.sum(label, 0)
                # print(utils.decode_sequence_with_len(model.vocab, key_word_list)[0])
                # print(key_word_list)
            # print(int(Key[Value.index("UNK")]))
            # attack mode 12: key_word_disappear + maintain original information, this mode only store NN of the original sents
            if opt.unseen_attack:
                key_word_list, _ = self.get_NN(model)
                key_word_list = torch.tensor(key_word_list)
                key_word_list = EUA.onehot_2_label(key_word_list)

                untar = tar_labels[1:seq_len]
                robust_set = torch.tensor([]).cuda()
                for i in range(len(untar)):
                    # print("key_word_disappear: %s" % opt.key_word_disappear)
                    # print(torch.argmax(untar[i]))
                    # print(torch.argmax(key_word_disappear_label,1))
                    # print(torch.argmax(key_word_list,1))
                    if torch.argmax(untar[i]) not in torch.argmax(key_word_disappear_label, 1) and torch.argmax(
                            untar[i]) in torch.argmax(key_word_list, 1):
                        robust_set = torch.cat((untar[i].unsqueeze(0), robust_set), 0)
                        # print(torch.argmax(untar[i]))
                # clamp: weights should not introduced in this mode
                robust_set = torch.clamp(robust_set, 0, 1)
                label = torch.cat((label.unsqueeze(0), -robust_set), 0)
            # attack mode 11: untargeted keyword disappear + main rest information
            if opt.ukd_mri:
                label = torch.sum(key_word_disappear_label, 0)
                untar = tar_labels[:seq_len].clone().detach()
                init_zeros = torch.zeros_like(untar[0])
                for i in range(len(untar)):
                    # print( torch.sum(untar[i] * label))
                    if torch.sum(untar[i] * label) >= 1:
                        # index = i
                        # rep_index = self.found_sub_label(model,untar,inputs,index,key_word_list)
                        # init_zeros[rep_index] = 1
                        untar[i] = init_zeros
                    else:
                        pass
                # untar[1] = init_zeros
                # attack mode 5: replace keyword_disappear using the given word --replace word
                if opt.replace_word:
                    re_label = self.process_single_label(opt,opt.replace_word)
                    untar = tar_labels[:seq_len].clone().detach()
                    for i in range(len(untar)):
                        if torch.sum(untar[i] * label) >= 1:
                            # index = i
                            # rep_index = self.found_sub_label(model,untar,inputs,index,key_word_list)
                            # init_zeros[rep_index] = 1
                            untar[i] = re_label
                        else:
                            pass
                    # untar[1] = init_zeros
                label = torch.cat((label.unsqueeze(0), -untar), 0)
        return label

    def _attack(self, model: nn.Module, optimizer: optim.Optimizer, inputs: torch.Tensor,
                modifier: torch.Tensor, label: torch.Tensor,
                const: torch.Tensor, seq_len: torch.Tensor, opt) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(inputs.shape) == 3:
            batch_size = 1
        elif len(inputs.shape) == 4:
            batch_size = inputs.shape[0]
        self.sub_div = self.sub_div.to(self.device)
        self.sub_init = self.sub_init.to(self.device)

        inputs = inputs * self.sub_div + self.sub_init
        tinputs = self._arctanh((inputs - self.boxplus) / self.boxmul)

        for iteration in range(self.max_iterations):
            # perform the attack

            adv_input = torch.tanh(tinputs + modifier) * self.boxmul + self.boxplus
            l2 = (adv_input - inputs).reshape(batch_size, -1).pow(2)
            adv_input_sub = (adv_input - self.sub_init) / self.sub_div
            data = self.get_feature(adv_input_sub)

            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
            tmp = [_.to('cuda') if _ is not None else _ for _ in tmp]
            fc_feats, att_feats, _, masks, att_masks = tmp
            # forward the model to also get generated samples for each image

            tmp_eval_kwargs = self.eval_kwargs.copy()
            tmp_eval_kwargs.update({'sample_n': 1})
            seq, logits = model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')

            if opt.targeted:
                if not opt.sup_word:
                    logit_dists = label * logits.squeeze(0)[:seq_len]
                if opt.targeted_key_word:
                    logit_dists = 2*torch.min(torch.sum(label*logits.squeeze(0)[:seq_len],1),0)[0]- torch.sum(torch.sum(label*logits.squeeze(0)[:seq_len],1),0)/seq_len
                    
                    #logit_dists = torch.min(torch.max(  -(1+label)*logits.squeeze(0)[:seq_len],1 ) - torch.sum( label*logits.squeeze(0)[:seq_len],1),0)
                    
                    
                    #logit_dists = label * torch.sum(logits.squeeze(0)[:],0)
                if opt.sup_word:
                    sup_label = -self.process_single_label(opt,opt.sup_word)
                    logit_dists = len(label)*torch.min(torch.sum(sup_label*logits.squeeze(0)[:seq_len],1),0)[0]
                    split_logit = self.split_joint(logits.squeeze(0)[:seq_len])
                    #logit_dists = 0 
                    for i in range(len(label)):
                        #logit_dists += torch.min(torch.sum(label[i]*split_logit,(1,2)),0)[0]
                        #logit_dists += torch.sum(label[i]*split_logit[i])
                        logit_dists += torch.min(torch.sum(label[i]*split_logit,(1,2)),0)[0]
            else:
                if opt.unseen_attack:
                    disappear = label[0]
                    logit_dists = 0
                    for i in range(1, len(label)):
                        logit_dists += torch.max(torch.sum(label[i] * logits.squeeze(0)[1:seq_len], 1))
                    logit_dists += (disappear * logits.squeeze(0)[1:seq_len]).sum() / (1. * seq_len)
                elif opt.ukd_mri:
                    disappear = label[0]
                    maintain = label[1:]
                    assert len(label[1:]) == len(logits.squeeze(0)[:seq_len])
                    logit_dists = (maintain * logits.squeeze(0)[
                                              :seq_len]).sum()  # -(maintain * logits.squeeze(0)[1]).sum()
                    logit_dists += (disappear * logits.squeeze(0)[1:seq_len]).sum() / (4. * seq_len)
                elif opt.key_word_disappear: 
                    logit_dists = (label * logits.squeeze(0)[1:seq_len]).max(0)[0]
                    #print(logit_dists)
                else:
                    logit_dists = label * logits.squeeze(0)[1:seq_len]

            if self.callback and (iteration + 1) % self.log_interval == 0:
                self.callback.scalar('logit_dist_{}'.format(outer_step), iteration + 1, logit_dists.mean().item())
                self.callback.scalar('l2_norm_{}'.format(outer_step), iteration + 1, l2.sqrt().mean().item())

            logit_loss = logit_dists.sum()
            l2_loss = l2.sum()
            loss = logit_loss# + l2_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            seq = seq.data
        #print("Noise Level:", l2.mean())
        noise = ((adv_input - inputs).pow(2).sqrt())*255.
        adv_input = adv_input*255.
        return seq.clone().detach(), logits.clone().detach(), adv_input_sub.clone().detach(), l2.sum().sqrt().clone().detach(),noise.clone().detach(),adv_input.clone().detach()

    def _eval(self, model: nn.Module, inputs: torch.Tensor):

        batch_size = inputs.shape[0]
        self.sub_div = self.sub_div.to(self.device)
        self.sub_init = self.sub_init.to(self.device)
        inputs = inputs * self.sub_div + self.sub_init

        data = self.get_feature(inputs)

        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
        tmp = [_.to('cuda') if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, labels, masks, att_masks = tmp
        # forward the model to also get generated samples for each image

        tmp_eval_kwargs = self.eval_kwargs.copy()
        tmp_eval_kwargs.update({'sample_n': 1})
        seq, logits = model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')

        return seq, logits

    def get_feature(self, imgs):
        # pick an index of the datapoint to load next

        if len(imgs.shape) == 3:
            batch_size = 1
            imgs = imgs.unsqueeze(0)
        elif len(imgs.shape) == 4:
            batch_size = imgs.shape[0]
        for i in range(batch_size):
            tmp_fc, tmp_att = self.my_resnet(imgs[i])
            if i == 0:
                tmp_fc_cat = tmp_fc
                tmp_att_cat = tmp_att
            else:
                tmp_fc_cat = torch.cat([tmp_fc_cat, tmp_fc], 0)
                tmp_att_cat = torch.cat([tmp_att_cat, tmp_att], 0)

        data = {}
        data['fc_feats'] = tmp_fc_cat.reshape(batch_size, 2048)
        data['att_feats'] = tmp_att_cat.reshape(batch_size, -1, 2048)
        data['labels'] = np.zeros([batch_size, 0])
        data['masks'] = None
        data['att_masks'] = None
        data = {k: torch.from_numpy(v) if type(v) is np.ndarray else v for k, v in
                data.items()}  # Turn all ndarray to torch tensor
        return data

    def attack(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, potential_set: torch.Tensor,
               seq_len: torch.Tensor,
               opt) -> torch.Tensor:
        """
        Performs the attack of the model for the inputs and labels.
        Parameters
        ----------
        model : nn.Module
            Model to attack.
        inputs : torch.Tensor
            Batch of samples to attack.
        labels : torch.Tensor
            Labels of the samples to attack if untargeted, else labels of targets.
        targeted : bool, optional
            Whether to perform a targeted attack or not.
        Returns
        -------
        torch.Tensor
            Batch of samples modified to be adversarial to the model
        """
        batch_size = inputs.shape[0]

        # set the lower and upper bounds accordingly
        lower_bound = torch.tensor(0, device=self.device)
        CONST = torch.tensor(self.initial_const, device=self.device)
        upper_bound = torch.tensor(1e10, device=self.device)

        # setup the target variable, we need it to be in one-hot form for the loss function
        label = self.define_label(labels.clone().detach(), potential_set, inputs, model, seq_len, opt)
        for outer_step in range(self.binary_search_steps):
            print("CONST:", CONST)
            modifier = torch.zeros_like(inputs, requires_grad=True)
            # setup the optimizer
            optimizer = optim.Adam([modifier], lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8)

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and outer_step == (self.binary_search_steps - 1):
                CONST = upper_bound
            seq, logits, adv,l2_noise,noise,adv_print = self._attack(model, optimizer, inputs, modifier, label, CONST, seq_len, opt)
            if opt.targeted_partical:
                label = self.define_label(labels, potential_set, adv, model, seq_len, opt, True)
                seq, logits, adv,l2_noise,noise,adv_print = self._attack(model, optimizer, inputs, modifier, label, CONST, seq_len, opt)

        if lower_bound < CONST * 10 and CONST * 10 < upper_bound:
            CONST *= 10
        return seq.clone().detach(), logits.clone().detach(),l2_noise.clone().detach(),noise.clone().detach(),adv_print.clone().detach()

    def _quantize(self, model: nn.Module, inputs: torch.Tensor, adv: torch.Tensor, labels: torch.Tensor,
                  targeted: bool = False) -> torch.Tensor:
        """
        Quantize the continuous adversarial inputs.
        model : nn.Module
            Model to attack.
        inputs : torch.Tensor
            Batch of samples to attack.
        adv : torch.Tensor
            Batch of continuous adversarial perturbations produced by the attack.
        labels : torch.Tensor
            Labels of the samples if untargeted, else labels of targets.
        targeted : bool, optional
            Whether to perform a targeted attack or not.
        Returns
        -------
        torch.Tensor
            Batch of samples modified to be quantized and adversarial to the model.
        """
        batch_size = inputs.shape[0]
        multiplier = 1 if targeted else -1
        delta = torch.round((adv - inputs) * 255) / 255
        delta.requires_grad_(True)
        logits = model(inputs + delta)
        is_adv = (logits.argmax(1) == labels) if targeted else (logits.argmax(1) != labels)
        i = 0
        while not is_adv.all() and i < 100:
            loss = F.cross_entropy(logits, labels, reduction='sum')
            grad = autograd.grad(loss, delta)[0].view(batch_size, -1)
            order = grad.abs().max(1, keepdim=True)[0]
            direction = (grad / order).int().float()
            direction.mul_(1 - is_adv.float().unsqueeze(1))
            delta.data.view(batch_size, -1).sub_(multiplier * direction / 255)

            logits = model(inputs + delta)
            is_adv = (logits.argmax(1) == labels) if targeted else (logits.argmax(1) != labels)
            i += 1

        delta.detach_()
        if not is_adv.all():
            delta.data[~is_adv].copy_(torch.round((adv[~is_adv] - inputs[~is_adv]) * 255) / 255)

        return inputs + delta

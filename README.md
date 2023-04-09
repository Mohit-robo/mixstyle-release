# MixStyle

This repo contains the code of our ICLR'21 paper, "Domain Generalization with MixStyle".

We have a medium [blog](https://medium.com/@mohit_gaikwad/review-mix-style-neural-networks-for-domain-generalization-and-adaptation-2207ece76707) reviewing the Mix-Style technique. 

Compared to the original [repo](https://github.com/KaiyangZhou/mixstyle-release), this implementation is only for **reid**. The changes are mentiioned below.     

**########## Updates ############**

**[28-06-2021]** A new implementation of MixStyle is out, which merges `MixStyle2` to `MixStyle` and switches between random and cross-domain mixing using `self.mix`. The new features can be found [here](https://github.com/KaiyangZhou/Dassl.pytorch/issues/23).

**[12-04-2021]** A variable `self._activated` is added to MixStyle to better control the computational flow. To deactivate MixStyle without modifying the model code, one can do
```python
def deactivate_mixstyle(m):
    if type(m) == MixStyle:
        m.set_activation_status(False)

model.apply(deactivate_mixstyle)
```
Similarly, to activate MixStyle, one can do
```python
def activate_mixstyle(m):
    if type(m) == MixStyle:
        m.set_activation_status(True)

model.apply(activate_mixstyle)
```
Note that `MixStyle` has been included in [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch). See [the code](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/modeling/backbone/resnet.py#L280) for details.

**##############################**

**A brief introduction**: The key idea of MixStyle is to probablistically mix instance-level feature statistics of training samples across source domains. MixStyle improves model robustness to domain shift by implicitly synthesizing new domains at the feature level for regularizing the training of convolutional neural networks. This idea is largely inspired by [neural style transfer](https://arxiv.org/abs/1703.06868) which has shown that feature statistics are closely related to image style and therefore arbitrary image style transfer can be achieved by switching the feature statistics between a content and a style image.

MixStyle is very easy to implement. Below we show a brief implementation of it in PyTorch. The full code can be found [here](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/modeling/ops/mixstyle.py).

```python
import random
import torch
import torch.nn as nn


class MixStyle(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1) # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(B // 2)]
            perm_a = perm_a[torch.randperm(B // 2)]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)

        return x_normed*sig_mix + mu_mix
```

How to apply MixStyle to your CNN models? Say you are using ResNet as the CNN architecture, and want to apply MixStyle after the 1st and 2nd residual blocks, you can first instantiate the MixStyle module using
```python
self.mixstyle = MixStyle(p=0.5, alpha=0.1)
```
during network construction (in `__init__()`), and then apply MixStyle in the forward pass like
```python
def forward(self, x):
    x = self.conv1(x) # 1st convolution layer
    x = self.res1(x) # 1st residual block
    x = self.mixstyle(x)
    x = self.res2(x) # 2nd residual block
    x = self.mixstyle(x)
    x = self.res3(x) # 3rd residual block
    x = self.res4(x) # 4th residual block
    ...
```

In our paper, we have demonstrated the effectiveness of MixStyle on three tasks: image classification, person re-identification, and reinforcement learning. The source code for reproducing all experiments can be found in `mixstyle-release/imcls`, `mixstyle-release/reid`, and `mixstyle-release/rl`, respectively.

**Changes made in this repo:** There is feature fusion performed, as compared to normal feature extraction from images. The idea has been taken from this AICIty 2020 Challenge [repo](https://github.com/layumi/AICIty-reID-2020/blob/677f3e46a8bd46a349b303a9497397c7e4e315a0/pytorch/test_2020.py#L191). The code changes made for feature extraction are as below, this block has to be replaced with [this](https://github.com/KaiyangZhou/deep-person-reid/blob/566a56a2cb255f59ba75aa817032621784df546a/torchreid/engine/engine.py#L361) function in the `torchreid` framework, i.e the `_evaluate` function and then perform the `torchreid` framework setup.

```python
def _feature_extraction(dataloader):
    
    features = torch.FloatTensor()
    pids_, camids_ = [], []

    for batch_idx, data in enumerate(dataloader):
        
        imgs, pids, camids = self.parse_data_for_eval(data)
        n, c, h, w = imgs.size()
        
        end = time.time()
        batch_time.update(time.time() - end)
        
        pids_.extend(pids.tolist())
        camids_.extend(camids.tolist())

        ff = torch.FloatTensor(n,2048).zero_().cuda()

        for i in range(2):
            if(i==1):
                imgs = fliplr(imgs)                
            
            outputs = self.extract_features(imgs.cuda())
            
            ff += outputs
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        
        features = torch.cat((features,ff.data.cpu().float()), 0)
    pids_ = np.asarray(pids_)
    camids_ = np.asarray(camids_)

    return features, pids_, camids_
```

The improved results have been mentioned in the table below:


| Model          |          | Original ||||Fine-tuned|||  
|----------------|----------|--------|--------|---------|------|------|---|-|
|ResNet50 Vanilla| **mAP**  | **R1** | **R5** | **R10** | **mAP** | **R1** | **R5** | **R10** |
|                |   19.3   | 35.4   |50.3    |56.4     |  20.8   | 38.1   | | |
|ResNet50 (mix-style)| **mAP**  | **R1** | **R5** | **R10** | **mAP** | **R1** | **R5** | **R10** |
|                |   23.8   | 42.2   |58.8    |64.8     |      


**Takeaways** on how to apply MixStyle to your tasks:
- Applying MixStyle to multiple lower layers is recommended (e.g., insert MixStyle after `res1` and `res2` in ResNets).
- Do not apply MixStyle to the last layer that is the closest to the prediction layer.
- Different tasks might favor different combinations.
- If you want to use the same configuration for all tasks/datasets for fair comparison, we suggest adding MixStyle to two consecutive layers, such as `res1` and `res2` in ResNets.

**To be done:** More experiments with feature fusion. 

For more analytical studies, please read our paper at https://openreview.net/forum?id=6xHJ37MVxxp.

Please also read the extended paper at https://arxiv.org/abs/2107.02053 for a more comprenehsive picture of MixStyle.

**Citations**

```
@inproceedings{zhou2021mixstyle,
  title={Domain Generalization with MixStyle},
  author={Zhou, Kaiyang and Yang, Yongxin and Qiao, Yu and Xiang, Tao},
  booktitle={ICLR},
  year={2021}
}

@article{zhou2021mixstylenn,
  title={MixStyle Neural Networks for Domain Generalization and Adaptation},
  author={Zhou, Kaiyang and Yang, Yongxin and Qiao, Yu and Xiang, Tao},
  journal={arXiv:2107.02053},
  year={2021}
}

@inproceedings{zheng2020going,
  title={Going beyond real data: A robust visual representation for vehicle re-identification},
  author={Zheng, Zhedong and Jiang, Minyue and Wang, Zhigang and Wang, Jian and Bai, Zechen and Zhang, Xuanmeng and Yu, Xin and Tan, Xiao and Yang, Yi and Wen, Shilei and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={598--599},
  year={2020}
}
```

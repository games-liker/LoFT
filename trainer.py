import os
import json
import time
import datetime
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, ConcatDataset
from torchvision import transforms

from clip import clip
from timm.models.vision_transformer import vit_base_patch16_224, vit_base_patch16_384, vit_large_patch16_224

import datasets
from models import *

from utils.meter import AverageMeter, OOD_AverageMeter
from utils.samplers import DownSampler
from utils.losses import *
from utils.evaluator import Evaluator
from utils.templates import ZEROSHOT_TEMPLATES

import open_clip

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from contextlib import ExitStack

from datasets.cifar_ssl import DATASET_GETTERS

def load_clip_to_cpu(backbone_name, prec):
    backbone_name = backbone_name.lstrip("CLIP-")
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu").eval()

    model = clip.build_model(state_dict or model.state_dict())

    assert prec in ["fp16", "fp32", "amp"]
    if prec == "fp32" or prec == "amp":
        # CLIP's default precision is fp16
        model.float()

    return model


def load_vit_to_cpu(backbone_name, prec):
    if backbone_name == "IN21K-ViT-B/16":
        model = vit_base_patch16_224(pretrained=True).eval()
    elif backbone_name == "IN21K-ViT-B/16@384px":
        model = vit_base_patch16_384(pretrained=True).eval()
    elif backbone_name == "IN21K-ViT-L/16":
        model = vit_large_patch16_224(pretrained=True).eval()

    assert prec in ["fp16", "fp32", "amp"]
    if prec == "fp16":
        # ViT's default precision is fp32
        model.half()
    
    return model

def load_openclip_to_cpu(backbone_name, prec):
    if backbone_name == "OpenCLIP-ViT-B/16":
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained=os.path.expanduser("~/.cache/huggingface/hub/models--laion--CLIP-ViT-B-16-laion2B-s34B-b88K/open_clip_pytorch_model.bin"))
    
    model = model.eval()
    state_dict = None
    model = clip.build_model(state_dict or model.state_dict())

    assert prec in ["fp16", "fp32", "amp"]
    if prec == "fp32" or prec == "amp":
        # CLIP's default precision is fp16
        model.float()

    return model

def load_metaclip_to_cpu(backbone_name, prec):
    if backbone_name == "MetaCLIP-ViT-B/16":
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-16-quickgelu', pretrained=os.path.expanduser("~/.cache/meta/b16_400m.pt"))
    model = model.eval()
    state_dict = None
    # file = model.state_dict()
    # new_state_dict = {}
    # for k, v in model.state_dict():
    #     new_k = k.replace('text_model', 'transformer')
    model = clip.build_model(state_dict or model.state_dict())

    assert prec in ["fp16", "fp32", "amp"]
    if prec == "fp32" or prec == "amp":
        # CLIP's default precision is fp16
        model.float()

    return model

class Trainer:
    def __init__(self, cfg):

        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        elif cfg.gpu is None:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(cfg.gpu)
            self.device = torch.device("cuda:{}".format(cfg.gpu))

        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        self.evaluator = Evaluator(cfg, self.many_idxs, self.med_idxs, self.few_idxs)
        self._writer = None

    def build_data_loader(self):
        cfg = self.cfg
        root = cfg.root
        resolution = cfg.resolution
        expand = cfg.expand

        if cfg.backbone.startswith("CLIP") or cfg.backbone.startswith("OpenCLIP") or cfg.backbone.startswith("MetaCLIP"):
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        print("mean:", mean)
        print("std:", std)

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        transform_plain = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        if cfg.tte:
            if cfg.tte_mode == "fivecrop":
                transform_test = transforms.Compose([
                    transforms.Resize(resolution + expand),
                    transforms.FiveCrop(resolution),
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                    transforms.Normalize(mean, std),
                ])
            elif cfg.tte_mode == "tencrop":
                transform_test = transforms.Compose([
                    transforms.Resize(resolution + expand),
                    transforms.TenCrop(resolution),
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                    transforms.Normalize(mean, std),
                ])
            elif cfg.tte_mode == "randaug":
                _resize_and_flip = transforms.Compose([
                    transforms.RandomResizedCrop(resolution),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
                transform_test = transforms.Compose([
                    transforms.Lambda(lambda image: torch.stack([_resize_and_flip(image) for _ in range(cfg.randaug_times)])),
                    transforms.Normalize(mean, std),
                ])
        else:
            transform_test = transforms.Compose([
                transforms.Resize(resolution * 8 // 7),
                transforms.CenterCrop(resolution),
                transforms.Lambda(lambda crop: torch.stack([transforms.ToTensor()(crop)])),
                transforms.Normalize(mean, std),
            ])

        train_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_train)
        train_init_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_plain)
        train_test_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_test)
        test_dataset = getattr(datasets, cfg.dataset)(root, train=False, transform=transform_test)

        self.num_classes = train_dataset.num_classes
        self.cls_num_list = train_dataset.cls_num_list
        self.classnames = train_dataset.classnames

        if cfg.dataset in ["CIFAR100", "CIFAR100_IR10", "CIFAR100_IR50"]:
            split_cls_num_list = datasets.CIFAR100_IR100(root, train=True).cls_num_list
        else:
            split_cls_num_list = self.cls_num_list
        self.many_idxs = (np.array(split_cls_num_list) > 100).nonzero()[0]
        self.med_idxs = ((np.array(split_cls_num_list) >= 20) & (np.array(split_cls_num_list) <= 100)).nonzero()[0]
        self.few_idxs = (np.array(split_cls_num_list) < 20).nonzero()[0]

        if cfg.init_head == "1_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=1)
        elif cfg.init_head == "10_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=10)
        elif cfg.init_head == "100_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=100)
        else:
            init_sampler = None

        self.train_loader = DataLoader(train_dataset,
            batch_size=cfg.micro_batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True)

        self.train_init_loader = DataLoader(train_init_dataset,
            batch_size=64, sampler=init_sampler, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)

        self.train_test_loader = DataLoader(train_test_dataset,
            batch_size=64, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)

        self.test_loader = DataLoader(test_dataset,
            batch_size=64, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)
        
        assert cfg.batch_size % cfg.micro_batch_size == 0
        self.accum_step = cfg.batch_size // cfg.micro_batch_size

        print("Total training points:", sum(self.cls_num_list))


    def build_model(self):
        cfg = self.cfg
        classnames = self.classnames
        num_classes = len(classnames)

        print("Building model")
        if cfg.zero_shot:
            assert cfg.backbone.startswith("CLIP-ViT") or cfg.backbone.startswith("OpenCLIP-ViT") or cfg.backbone.startswith("MetaCLIP-ViT")
            print(f"Loading CLIP (backbone: {cfg.backbone})")
            if cfg.backbone.startswith("CLIP-ViT"):
                clip_model = load_clip_to_cpu(cfg.backbone, cfg.prec)
            elif cfg.backbone.startswith("OpenCLIP-ViT"):
                clip_model = load_openclip_to_cpu(cfg.backbone, cfg.prec)
            elif cfg.backbone.startswith("MetaCLIP-ViT"):
                clip_model = load_metaclip_to_cpu(cfg.backbone, cfg.prec)
            self.model = ZeroShotCLIP(clip_model)
            self.model.to(self.device)
            self.tuner = None
            self.head = None

            template = "a photo of a {}."
            prompts = self.get_tokenized_prompts(classnames, template)
            self.model.init_text_features(prompts)

        elif cfg.backbone.startswith("CLIP"):
            print(f"Loading CLIP (backbone: {cfg.backbone})")
            clip_model = load_clip_to_cpu(cfg.backbone, cfg.prec)
            self.model = PeftModelFromCLIP(cfg, clip_model, num_classes)
            self.model.to(self.device)
            self.tuner = self.model.tuner
            self.head = self.model.head
            # self.unlabeled_head = self.model.unlabeled_head

            

        elif cfg.backbone.startswith("IN21K-ViT"):
            print(f"Loading ViT (backbone: {cfg.backbone})")
            vit_model = load_vit_to_cpu(cfg.backbone, cfg.prec)
            self.model = PeftModelFromViT(cfg, vit_model, num_classes)
            self.model.to(self.device)
            self.tuner = self.model.tuner
            self.head = self.model.head
        
        elif cfg.backbone.startswith("Meta"):
            print(f"Loading MetaCLIP (backbone: {cfg.backbone})")
            clip_model = load_metaclip_to_cpu(cfg.backbone, cfg.prec)
            self.model = PeftModelFromCLIP(cfg, clip_model, num_classes)
            self.model.to(self.device)
            self.tuner = self.model.tuner
            self.head = self.model.head
        
        elif cfg.backbone.startswith("Open"):
            print(f"Loading OpenCLIP (backbone: {cfg.backbone})")
            clip_model = load_openclip_to_cpu(cfg.backbone, cfg.prec)
            self.model = PeftModelFromCLIP(cfg, clip_model, num_classes)
            self.model.to(self.device)
            self.tuner = self.model.tuner
            self.head = self.model.head

        if not (cfg.zero_shot or cfg.test_train or cfg.test_only):
            self.build_optimizer()
            self.build_criterion()

            if cfg.init_head == "text_feat":
                self.init_head_text_feat()
            elif cfg.init_head in ["class_mean", "1_shot", "10_shot", "100_shot"]:
                self.init_head_class_mean()
            elif cfg.init_head == "linear_probe":
                self.init_head_linear_probe()
            else:
                print("No initialization with head")
            
            torch.cuda.empty_cache()
        
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1 and cfg.gpu is None:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def build_optimizer(self):
        cfg = self.cfg

        print("Turning off gradients in the model")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
        print("Turning on gradients in the tuner")
        for name, param in self.tuner.named_parameters():
            param.requires_grad_(True)
        print("Turning on gradients in the head")
        for name, param in self.head.named_parameters():
            param.requires_grad_(True)
        # print("Turning on gradients in the unlabeled head")
        # for name, param in self.unlabeled_head.named_parameters():
        #     param.requires_grad_(True)

        # print parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        tuned_params = sum(p.numel() for p in self.tuner.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        # unlabeled_head_params = sum(p.numel() for p in self.unlabeled_head.parameters())
        print(f"Total params: {total_params}")
        print(f"Tuned params: {tuned_params}")
        print(f"Head params: {head_params}")
        # print(f"unlabeled Head params: {unlabeled_head_params}")

        # for name, param in self.tuner.named_parameters():
        #     print(name, param.numel())

        # NOTE: only give tuner and head to the optimizer
        self.optim = torch.optim.SGD([{"params": self.tuner.parameters()},
                                      {"params": self.head.parameters()}],
                                    #   {"params": self.unlabeled_head.parameters()}],
                                      lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, cfg.num_epochs)
        self.scaler = GradScaler() if cfg.prec == "amp" else None

    def build_criterion(self):
        cfg = self.cfg
        cls_num_list = torch.Tensor(self.cls_num_list).to(self.device)

        if cfg.loss_type == "CE":
            self.criterion = nn.CrossEntropyLoss()
        elif cfg.loss_type == "Focal": # https://arxiv.org/abs/1708.02002
            self.criterion = FocalLoss()
        elif cfg.loss_type == "LDAM": # https://arxiv.org/abs/1906.07413
            self.criterion = LDAMLoss(cls_num_list=cls_num_list, s=cfg.scale)
        elif cfg.loss_type == "CB": # https://arxiv.org/abs/1901.05555
            self.criterion = ClassBalancedLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "GRW": # https://arxiv.org/abs/2103.16370
            self.criterion = GeneralizedReweightLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "BS": # https://arxiv.org/abs/2007.10740
            self.criterion == BalancedSoftmaxLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "LA": # https://arxiv.org/abs/2007.07314
            self.criterion = LogitAdjustedLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "LADE": # https://arxiv.org/abs/2012.00321
            self.criterion = LADELoss(cls_num_list=cls_num_list)
        
    def get_tokenized_prompts(self, classnames, template):
        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        # print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)
        return prompts

    @torch.no_grad()
    def init_head_text_feat(self):
        cfg = self.cfg
        classnames = self.classnames

        print("Initialize head with text features")
        if cfg.prompt == "ensemble":
            all_text_features = []
            for template in tqdm(ZEROSHOT_TEMPLATES['imagenet']):
                prompts = self.get_tokenized_prompts(classnames, template)
                text_features = self.model.encode_text(prompts)
                text_features = F.normalize(text_features, dim=-1)
                all_text_features.append(text_features)
            all_text_features = torch.stack(all_text_features)
            text_features = all_text_features.mean(dim=0)
        elif cfg.prompt == "descriptor":
            with open("utils/descriptors_imagenet.json") as f:
                descriptors = json.load(f)
            template = "{}"
            all_class_features = []
            for cn in tqdm(classnames):
                prompts = self.get_tokenized_prompts(descriptors[cn], template)
                text_features = self.model.encode_text(prompts)
                text_features = F.normalize(text_features, dim=-1)
                all_class_features.append(text_features.mean(dim=0))
            text_features = torch.stack(all_class_features)
        elif cfg.prompt == "classname":
            template = "{}"
            prompts = self.get_tokenized_prompts(classnames, template)
            text_features = self.model.encode_text(prompts)
            text_features = F.normalize(text_features, dim=-1)
        elif cfg.prompt == "default":
            template = "a photo of a {}."
            prompts = self.get_tokenized_prompts(classnames, template)
            text_features = self.model.encode_text(prompts)
            text_features = F.normalize(text_features, dim=-1)

        if cfg.backbone.startswith("CLIP-ViT") or cfg.backbone.startswith("OpenCLIP-ViT") or cfg.backbone.startswith("MetaCLIP-ViT"):
            text_features = text_features @ self.model.image_encoder.proj.t()
            text_features = F.normalize(text_features, dim=-1)
        

        self.head.apply_weight(text_features)
        # self.unlabeled_head.apply_weight(text_features)

    @torch.no_grad()
    def init_head_class_mean(self):
        print("Initialize head with class means")
        all_features = []
        all_labels = []

        for batch in tqdm(self.train_init_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            feature = self.model(image, use_tuner=False, return_feature=True)

            all_features.append(feature)
            all_labels.append(label)

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        sorted_index = all_labels.argsort()
        all_features = all_features[sorted_index]
        all_labels = all_labels[sorted_index]

        unique_labels, label_counts = torch.unique(all_labels, return_counts=True)

        class_means = [None] * self.num_classes
        idx = 0
        for i, cnt in zip(unique_labels, label_counts):
            class_means[i] = all_features[idx: idx+cnt].mean(dim=0, keepdim=True)
            idx += cnt
        class_means = torch.cat(class_means, dim=0)
        class_means = F.normalize(class_means, dim=-1)

        self.head.apply_weight(class_means)

    @torch.no_grad()
    def init_head_linear_probe(self):
        print("Initialize head with linear probing")
        all_features = []
        all_labels = []

        for batch in tqdm(self.train_init_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            feature = self.model(image, use_tuner=False, return_feature=True)

            all_features.append(feature)
            all_labels.append(label)

        all_features = torch.cat(all_features, dim=0).cpu()
        all_labels = torch.cat(all_labels, dim=0).cpu()

        clf = LogisticRegression(solver="lbfgs", max_iter=100, penalty="l2", class_weight="balanced").fit(all_features, all_labels)
        class_weights = torch.from_numpy(clf.coef_).to(all_features.dtype).to(self.device)
        class_weights = F.normalize(class_weights, dim=-1)

        self.head.apply_weight(class_weights)

    def train(self):
        cfg = self.cfg

        # Initialize summary writer
        writer_dir = os.path.join(cfg.output_dir, "tensorboard")
        os.makedirs(writer_dir, exist_ok=True)
        print(f"Initialize tensorboard (log_dir={writer_dir})")
        self._writer = SummaryWriter(log_dir=writer_dir)

        # Initialize average meters
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter(ema=True)
        acc_meter = AverageMeter(ema=True)
        cls_meters = [AverageMeter(ema=True) for _ in range(self.num_classes)]

        # Remember the starting time (for computing the elapsed time)
        time_start = time.time()

        num_epochs = cfg.num_epochs
        for epoch_idx in range(num_epochs):
            self.tuner.train()
            end = time.time()

            num_batches = len(self.train_loader)
            for batch_idx, batch in enumerate(self.train_loader):
                data_time.update(time.time() - end)

                image = batch[0]
                label = batch[1]
                image = image.to(self.device)
                label = label.to(self.device)

                if cfg.prec == "amp":
                    with autocast():
                        output = self.model(image)
                        loss = self.criterion(output, label)
                        loss_micro = loss / self.accum_step
                        self.scaler.scale(loss_micro).backward()
                    if ((batch_idx + 1) % self.accum_step == 0) or (batch_idx + 1 == num_batches):
                        self.scaler.step(self.optim)
                        self.scaler.update()
                        self.optim.zero_grad()
                else:
                    output = self.model(image)
                    loss = self.criterion(output, label)
                    loss_micro = loss / self.accum_step
                    loss_micro.backward()
                    if ((batch_idx + 1) % self.accum_step == 0) or (batch_idx + 1 == num_batches):
                        self.optim.step()
                        self.optim.zero_grad()

                with torch.no_grad():
                    pred = output.argmax(dim=1)
                    correct = pred.eq(label).float()
                    acc = correct.mean().mul_(100.0)

                current_lr = self.optim.param_groups[0]["lr"]
                loss_meter.update(loss.item())
                acc_meter.update(acc.item())
                batch_time.update(time.time() - end)

                for _c, _y in zip(correct, label):
                    cls_meters[_y].update(_c.mul_(100.0).item(), n=1)
                cls_accs = [cls_meters[i].avg for i in range(self.num_classes)]

                mean_acc = np.mean(np.array(cls_accs))
                many_acc = np.mean(np.array(cls_accs)[self.many_idxs])
                med_acc = np.mean(np.array(cls_accs)[self.med_idxs])
                few_acc = np.mean(np.array(cls_accs)[self.few_idxs])

                meet_freq = (batch_idx + 1) % cfg.print_freq == 0
                only_few_batches = num_batches < cfg.print_freq
                if meet_freq or only_few_batches:
                    nb_remain = 0
                    nb_remain += num_batches - batch_idx - 1
                    nb_remain += (
                        num_epochs - epoch_idx - 1
                    ) * num_batches
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                    info = []
                    info += [f"epoch [{epoch_idx + 1}/{num_epochs}]"]
                    info += [f"batch [{batch_idx + 1}/{num_batches}]"]
                    info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                    info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                    info += [f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]
                    info += [f"acc {acc_meter.val:.4f} ({acc_meter.avg:.4f})"]
                    info += [f"(mean {mean_acc:.4f} many {many_acc:.4f} med {med_acc:.4f} few {few_acc:.4f})"]
                    info += [f"lr {current_lr:.4e}"]
                    info += [f"eta {eta}"]
                    print(" ".join(info))

                n_iter = epoch_idx * num_batches + batch_idx
                self._writer.add_scalar("train/lr", current_lr, n_iter)
                self._writer.add_scalar("train/loss.val", loss_meter.val, n_iter)
                self._writer.add_scalar("train/loss.avg", loss_meter.avg, n_iter)
                self._writer.add_scalar("train/acc.val", acc_meter.val, n_iter)
                self._writer.add_scalar("train/acc.avg", acc_meter.avg, n_iter)
                self._writer.add_scalar("train/mean_acc", mean_acc, n_iter)
                self._writer.add_scalar("train/many_acc", many_acc, n_iter)
                self._writer.add_scalar("train/med_acc", med_acc, n_iter)
                self._writer.add_scalar("train/few_acc", few_acc, n_iter)
                
                end = time.time()

            self.sched.step()
            torch.cuda.empty_cache()

        print("Finish training")
        print("Note that the printed training acc is not precise.",
              "To get precise training acc, use option ``test_train True``.")

        # show elapsed time
        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Time elapsed: {elapsed}")

        # save model
        self.save_model(cfg.output_dir)

        self.test()

        # Close writer
        self._writer.close()

    @torch.no_grad()
    def test(self, mode="test"):
        if self.tuner is not None:
            self.tuner.eval()
        if self.head is not None:
            self.head.eval()
        self.evaluator.reset()

        if mode == "train":
            print(f"Evaluate on the train set")
            data_loader = self.train_test_loader
        elif mode == "test":
            print(f"Evaluate on the test set")
            data_loader = self.test_loader

        for batch in tqdm(data_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            _bsz, _ncrops, _c, _h, _w = image.size()
            image = image.view(_bsz * _ncrops, _c, _h, _w)

            if _ncrops <= 5:
                output = self.model(image)
                output = output.view(_bsz, _ncrops, -1).mean(dim=1)
            else:
                # CUDA out of memory
                output = []
                image = image.view(_bsz, _ncrops, _c, _h, _w)
                for k in range(_ncrops):
                    output.append(self.model(image[:, k]))
                output = torch.stack(output).mean(dim=0)

            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"test/{k}"
            if self._writer is not None:
                self._writer.add_scalar(tag, v)

        return list(results.values())[0]
    

    def shot_acc(self, preds, labels, train_class_count, acc_per_cls=False):
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
        elif isinstance(preds, np.ndarray):
            pass
        else:
            raise TypeError('Type ({}) of preds not supported'.format(type(preds)))

        num_classes = len(train_class_count)

        test_class_count = [np.nan] * num_classes
        class_correct = [np.nan] * num_classes
        for l in range(num_classes):
            test_class_count[l] = len(labels[labels == l])
            class_correct[l] = (preds[labels == l] == labels[labels == l]).sum()

        if num_classes <= 100: # e.g. On CIFAR10/100
            many_shot_thr = train_class_count[int(0.34*num_classes)]
            low_shot_thr = train_class_count[int(0.67*num_classes)]
        else:
            many_shot_thr=100
            low_shot_thr=20
        # print(many_shot_thr, low_shot_thr)

        many_shot = []
        median_shot = []
        low_shot = []
        for i in range(num_classes):
            if test_class_count[i] == 0:
                assert class_correct[i] == 0
                _acc_class_i = np.nan
            else:
                _acc_class_i = class_correct[i] / test_class_count[i]
            if train_class_count[i] > many_shot_thr:
                many_shot.append(_acc_class_i)
            elif train_class_count[i] < low_shot_thr:
                low_shot.append(_acc_class_i)
            else:
                median_shot.append(_acc_class_i)    

        # print('many_shot:', many_shot)
        # print('median_shot:', median_shot)
        # print('low_shot:', low_shot)

        if acc_per_cls:
            class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)] 
            return np.nanmean(many_shot), np.nanmean(median_shot), np.nanmean(low_shot), class_accs
        else:
            return np.nanmean(many_shot), np.nanmean(median_shot), np.nanmean(low_shot)
    
    def stable_cumsum(self, arr, rtol=1e-05, atol=1e-08):
        """Use high precision for cumsum and check that final value matches sum
        Parameters
        ----------
        arr : array-like
            To be cumulatively summed as flat
        rtol : float
            Relative tolerance, see ``np.allclose``
        atol : float
            Absolute tolerance, see ``np.allclose``
        """
        out = np.cumsum(arr, dtype=np.float64)
        expected = np.sum(arr, dtype=np.float64)
        if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
            raise RuntimeError('cumsum was found to be unstable: '
                            'its last element does not correspond to sum')
        return out
    
    def fpr_and_fdr_at_recall(self, y_true, y_score, recall_level=0.95, pos_label=None):
        classes = np.unique(y_true)
        if (pos_label is None and
                not (np.array_equal(classes, [0, 1]) or
                        np.array_equal(classes, [-1, 1]) or
                        np.array_equal(classes, [0]) or
                        np.array_equal(classes, [-1]) or
                        np.array_equal(classes, [1]))):
            raise ValueError("Data is not binary and pos_label is not specified")
        elif pos_label is None:
            pos_label = 1.

        # make y_true a boolean vector
        y_true = (y_true == pos_label)

        # sort scores and corresponding truth values
        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
        y_score = y_score[desc_score_indices]
        y_true = y_true[desc_score_indices]

        # y_score typically has many tied values. Here we extract
        # the indices associated with the distinct values. We also
        # concatenate a value for the end of the curve.
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

        # accumulate the true positives with decreasing threshold
        tps = self.stable_cumsum(y_true)[threshold_idxs]
        fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

        thresholds = y_score[threshold_idxs]

        recall = tps / tps[-1]

        last_ind = tps.searchsorted(tps[-1])
        sl = slice(last_ind, None, -1)      # [last_ind::-1]
        recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

        cutoff = np.argmin(np.abs(recall - recall_level))

        return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])

    
    def get_measures(self, _pos, _neg, recall_level=0.95):
        pos = np.array(_pos[:]).reshape((-1, 1))
        neg = np.array(_neg[:]).reshape((-1, 1))
        examples = np.squeeze(np.vstack((pos, neg)))
        labels = np.zeros(len(examples), dtype=np.int32)
        labels[:len(pos)] += 1
        auroc = roc_auc_score(labels, examples)
        aupr_in = average_precision_score(labels, examples)
        labels_rev = np.zeros(len(examples), dtype=np.int32)
        labels_rev[len(pos):] += 1
        aupr_out = average_precision_score(labels_rev, -examples)
        fpr = self.fpr_and_fdr_at_recall(labels, examples, recall_level)

        return auroc, aupr_in, aupr_out, fpr, pos.mean(), neg.mean()
    
    @torch.no_grad()
    def ood_test(self, mode="ood_test", metric='msp', dout='cifar'):
        if self.tuner is not None:
            self.tuner.eval()
        if self.head is not None:
            self.head.eval()
        self.evaluator.reset()

        if mode == "train":
            print(f"Evaluate on the train set")
            data_loader = self.train_test_loader
        elif mode == "test":
            print(f"Evaluate on the test set")
            data_loader = self.test_loader
        elif mode == "ood_test":
            print(f"Evaluate on the test and ood set")
            test_loader = self.test_loader
            ood_loader = self.ood_loader

        ts = time.time()
        test_acc_meter = OOD_AverageMeter()
        score_list = []
        labels_list = []
        pred_list = []
        probs_list = []
        with ExitStack() as stack:
            if metric not in ['odin', 'maha']:
                stack.enter_context(torch.no_grad())

            for images, targets in test_loader:
                images, targets = images.cuda(), targets.cuda()

                _bsz, _ncrops, _c, _h, _w = images.size()
                images = images.view(_bsz * _ncrops, _c, _h, _w)

                logits = self.model(images)
                probs = F.softmax(logits, dim=1)
                msp = probs.max(dim=1).values
                scores = - msp # The larger MSP, the smaller uncertainty
                # logits, scores = get_scores_fn(model, images)
                # probs = F.softmax(logits, dim=1)
                pred = logits.data.max(1)[1]
                acc = pred.eq(targets.data).float().mean()
                # append loss:
                score_list.append(scores.detach().cpu().numpy())
                labels_list.append(targets.detach().cpu().numpy())
                pred_list.append(pred.detach().cpu().numpy())
                probs_list.append(probs.max(dim=1).values.detach().cpu().numpy())
                test_acc_meter.append(acc.item())
        print('clean test time: %.2fs' % (time.time()-ts))
        # test loss and acc of this epoch:
        test_acc = test_acc_meter.avg
        in_scores = np.concatenate(score_list, axis=0)
        in_labels = np.concatenate(labels_list, axis=0)
        in_preds = np.concatenate(pred_list, axis=0)

        img_num_per_cls = self.cls_num_list

        many_acc, median_acc, low_acc, _ = self.shot_acc(in_preds, in_labels, img_num_per_cls, acc_per_cls=True)

        clean_str = 'ACC: %.4f (%.4f, %.4f, %.4f)' % (test_acc, many_acc, median_acc, low_acc)
        print(clean_str)
        # fp.write(clean_str + '\n')
        # fp.flush()
        
        # confidence distribution of correct samples:
        ood_score_list, sc_labels_list = [], []
        with ExitStack() as stack:
            if metric not in ['odin', 'maha']:
                stack.enter_context(torch.no_grad())

            for images, sc_labels in ood_loader:
                images, sc_labels = images.cuda(), sc_labels.cuda()
                logits = self.model(images)
                probs = F.softmax(logits, dim=1)
                msp = probs.max(dim=1).values
                scores = - msp # The larger MSP, the smaller uncertainty
                # logits, scores = get_scores_fn(model, images)
                # append loss:
                ood_score_list.append(scores.detach().cpu().numpy())
                sc_labels_list.append(sc_labels.detach().cpu().numpy())
        ood_scores = np.concatenate(ood_score_list, axis=0)
        sc_labels = np.concatenate(sc_labels_list, axis=0)

        # if args.dout == 'svhn':
        #     np.save(os.path.join(save_dir, 'ood_scores.npy'), ood_scores)

        # move some elements in ood_scores to in_scores:
        print('in_scores:', in_scores.shape)
        print('ood_scores:', ood_scores.shape)
        fake_ood_scores = ood_scores[sc_labels>=0]
        real_ood_scores = ood_scores[sc_labels<0]
        real_in_scores = np.concatenate([in_scores, fake_ood_scores], axis=0)
        print('fake_ood_scores:', fake_ood_scores.shape)
        print('real_in_scores:', real_in_scores.shape)
        print('real_ood_scores:', real_ood_scores.shape)

        auroc, aupr_in, aupr_out, fpr95, id_meansocre, ood_meanscore = self.get_measures(-real_in_scores, -real_ood_scores)
        # print:
        ood_detectoin_str = 'auroc: %.4f, aupr_in: %.4f, aupr_out: %.4f, fpr95: %.4f, ood_meanscore: %.4f, id_meansocre: %.4f' % (auroc, aupr_in, aupr_out, fpr95, ood_meanscore, id_meansocre)
        print(ood_detectoin_str)
        # fp.write(ood_detectoin_str + '\n')
        # fp.flush()
        # fp.close()
        return 

    def save_model(self, directory):
        tuner_dict = self.tuner.state_dict()
        head_dict = self.head.state_dict()
        checkpoint = {
            "tuner": tuner_dict,
            "head": head_dict
        }

        # remove 'module.' in state_dict's keys
        for key in ["tuner", "head"]:
            state_dict = checkpoint[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]
                new_state_dict[k] = v
            checkpoint[key] = new_state_dict

        # save model
        save_path = os.path.join(directory, "checkpoint.pth.tar")
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    def load_model(self, directory):
        load_path = os.path.join(directory, "checkpoint.pth.tar")

        if not os.path.exists(load_path):
            raise FileNotFoundError('Checkpoint not found at "{}"'.format(load_path))

        checkpoint = torch.load(load_path, map_location=self.device)
        tuner_dict = checkpoint["tuner"]
        head_dict = checkpoint["head"]

        print("Loading weights to from {}".format(load_path))
        self.tuner.load_state_dict(tuner_dict, strict=False)

        if head_dict["weight"].shape == self.head.weight.shape:
            self.head.load_state_dict(head_dict, strict=False)


class SSL_Trainer(Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    
    def build_data_loader(self):
        cfg = self.cfg
        root = cfg.root
        resolution = cfg.resolution
        expand = cfg.expand

        if cfg.backbone.startswith("CLIP") or cfg.backbone.startswith("OpenCLIP") or cfg.backbone.startswith("MetaCLIP"):
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        print("mean:", mean)
        print("std:", std)
        
        # labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[cfg.dataset](
        #     cfg, 'datasets/'+cfg.dataset_name)

        # 根据cfg.dataset获取数据集名称,转换为小写以匹配DATASET_GETTERS的key
        dataset_name = cfg.dataset.lower()
        # 处理特殊的数据集名称映射
        if dataset_name.startswith('cifar100'):
            dataset_name = 'cifar100'
        elif dataset_name.startswith('cifar10'):
            dataset_name = 'cifar10'
        elif dataset_name.startswith('smallimagenet'):
            dataset_name = 'smallimagenet'
        elif dataset_name.startswith('place'):
            dataset_name = 'place'
        
        if dataset_name not in DATASET_GETTERS:
            raise ValueError(f"Dataset '{dataset_name}' not found in DATASET_GETTERS. Available: {list(DATASET_GETTERS.keys())}")
        
        labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[dataset_name](cfg, root)
        
        # 如果是Open模式,将unlabeled dataset与COCO dataset concat
        if getattr(cfg, 'is_open', False):
            # 获取OOD数据集配置
            ood_dataset_name = getattr(cfg, 'ood_dataset', 'coco')
            if ood_dataset_name is None:
                ood_dataset_name = 'coco'
            ood_dataset_name = ood_dataset_name.lower()
            
            # 获取OOD数据集根目录
            ood_root = getattr(cfg, 'ood_root', None)
            if ood_root is None:
                ood_root = "datasets/train2017"
                print(f"Warning: ood_root not specified in config, using default: {ood_root}")
            
            if ood_dataset_name not in DATASET_GETTERS:
                raise ValueError(f"OOD dataset '{ood_dataset_name}' not found in DATASET_GETTERS. Available: {list(DATASET_GETTERS.keys())}")
            
            # 加载OOD数据集(COCO)
            ood_dataset = DATASET_GETTERS[ood_dataset_name](cfg, ood_root)
            
            # 保存原始unlabeled dataset的大小
            original_unlabeled_size = len(unlabeled_dataset)
            
            # 将unlabeled dataset与COCO dataset concat
            unlabeled_dataset = ConcatDataset([unlabeled_dataset, ood_dataset])
            print(f"Open mode enabled: Concatenated unlabeled dataset with {ood_dataset_name} dataset")
            print(f"Unlabeled dataset size: {len(unlabeled_dataset)} (original: {original_unlabeled_size}, COCO: {len(ood_dataset)})")
        
        self.num_classes = labeled_dataset.num_classes
        self.cls_num_list = labeled_dataset.cls_num_list
        self.classnames = labeled_dataset.classnames
        # print(self.classnames)

        self.many_idxs = (np.array(self.cls_num_list) > 100).nonzero()[0]
        self.med_idxs = ((np.array(self.cls_num_list) >= 20) & (np.array(self.cls_num_list) <= 100)).nonzero()[0]
        self.few_idxs = (np.array(self.cls_num_list) < 20).nonzero()[0]

        train_sampler = RandomSampler if cfg.local_rank == -1 else DistributedSampler
        self.labeled_trainloader = DataLoader(
            labeled_dataset,
            sampler=train_sampler(labeled_dataset),
            batch_size=cfg.micro_batch_size,
            num_workers=cfg.num_workers,
            drop_last=True)

        self.unlabeled_trainloader = DataLoader(
            unlabeled_dataset,
            sampler=train_sampler(unlabeled_dataset),
            batch_size=cfg.micro_batch_size*cfg.mu,
            num_workers=cfg.num_workers,
            drop_last=True)

        self.test_loader = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            batch_size=cfg.micro_batch_size,
            num_workers=cfg.num_workers)
        
        assert cfg.batch_size % cfg.micro_batch_size == 0
        self.accum_step = cfg.batch_size // cfg.micro_batch_size

        print("Labeled training points:", sum(self.cls_num_list))
        # print("UnLabeled training points:", sum(unlabeled_dataset.cls_num_list))
        # print("UnLabeled training points:", len(unlabeled_dataset))

    
    def ssl_train(self):
        def interleave(x, size):
            s = list(x.shape)
            return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

        def de_interleave(x, size):
            s = list(x.shape)
            return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

        args = self.cfg
        cfg = self.cfg
        
        # Initialize summary writer
        writer_dir = os.path.join(cfg.output_dir, "tensorboard")
        os.makedirs(writer_dir, exist_ok=True)
        print(f"Initialize tensorboard (log_dir={writer_dir})")
        self._writer = SummaryWriter(log_dir=writer_dir)

        # Initialize average meters
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter(ema=True)
        acc_meter = AverageMeter(ema=True)
        cls_meters = [AverageMeter(ema=True) for _ in range(self.num_classes)]

        labeled_iter = iter(self.labeled_trainloader)
        unlabeled_iter = iter(self.unlabeled_trainloader)

        # Remember the starting time (for computing the elapsed time)
        time_start = time.time()
        args.epochs = math.ceil(args.total_steps / args.eval_step)
        num_epochs = args.epochs - args.start_epoch
        for epoch_idx in range(args.start_epoch, args.epochs):
            self.tuner.train()
            end = time.time()

            # num_batches = len(self.train_loader)
            # for batch_idx, batch in enumerate(self.train_loader):
            num_batches = args.eval_step
            for batch_idx in range(args.eval_step):
                try:
                    # inputs_x, targets_x = labeled_iter.next()
                    # error occurs ↓
                    inputs_x, targets_x = next(labeled_iter)
                except:
                    # if args.world_size > 1:
                    #     labeled_epoch += 1
                    #     self.labeled_trainloader.sampler.set_epoch(labeled_epoch)
                    labeled_iter = iter(self.labeled_trainloader)
                    # inputs_x, targets_x = labeled_iter.next()
                    # error occurs ↓
                    inputs_x, targets_x = next(labeled_iter)
                
                try:
                    # (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
                    # error occurs ↓
                    (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
                    # (inputs_u_w, inputs_u_s), unlabel = next(unlabeled_iter)
                    # inputs_u_w, inputs_u_s = next(unlabeled_iter)
                except:
                    # if args.world_size > 1:
                    #     unlabeled_epoch += 1
                    #     self.unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                    unlabeled_iter = iter(self.unlabeled_trainloader)
                    # (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
                    # error occurs ↓
                    (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
                    # (inputs_u_w, inputs_u_s), unlabel = next(unlabeled_iter)
                    # inputs_u_w, inputs_u_s = next(unlabeled_iter)
                data_time.update(time.time() - end)
                batch_size = inputs_x.shape[0]
                inputs = interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(self.device)
                targets_x = targets_x.to(self.device)
                # unlabel = unlabel.to(self.device)
                logits = self.model(inputs)
                logits = de_interleave(logits, 2*args.mu+1)
                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
                del logits
                # logits = self.model(torch.cat((inputs_x, inputs_u_w)).to(self.device))
                # logits_x = logits[:batch_size]
                # logits_u_w = logits[batch_size:]

                # inputs_u_s = inputs_u_s.to(self.device)
                # zero_shot_logits = self.model.zero_shot(inputs_u_s)
                # del logits
                # logits_u_s = self.model(inputs_u_s, unlabeled=True)

                # Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
                Lx = self.criterion(logits_x, targets_x)

                pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(args.threshold).float()
                ood_mask = max_probs.le(cfg.ood_threshold).float()
                    
                Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()
                Luc = (F.cross_entropy(logits_u_s, pseudo_label, reduction='none') * (1 - mask) * (1 - ood_mask)).mean()

                
                # eps = 1e-10
                # logits_u_s = torch.softmax(logits_u_s, dim=-1)
                # KLu = (torch.sum(pseudo_label * (torch.log(pseudo_label+eps) - torch.log(logits_u_s+eps)),
                                    # dim=-1) * mask).mean()

                # Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()
                # Lu = F.cross_entropy(logits_u_s, unlabel)
                # Luc = (F.cross_entropy(logits_u_s, pseudo_label, reduction='none') * (1 - mask)).mean()
                # Luc = 0

                loss = Lx + args.lambda_u * Lu + args.lambda_uc * Luc
                # loss = Lx + args.lambda_u * KLu

                if cfg.prec == "amp":
                    with autocast():
                        loss_micro = loss / self.accum_step
                        self.scaler.scale(loss_micro).backward()
                    if ((batch_idx + 1) % self.accum_step == 0) or (batch_idx + 1 == num_batches):
                        self.scaler.step(self.optim)
                        self.scaler.update()
                        self.optim.zero_grad()
                else:
                    loss_micro = loss / self.accum_step
                    loss_micro.backward()
                    if ((batch_idx + 1) % self.accum_step == 0) or (batch_idx + 1 == num_batches):
                        self.optim.step()
                        self.optim.zero_grad()

                # with torch.no_grad():
                #     pred = output.argmax(dim=1)
                #     correct = pred.eq(label).float()
                #     acc = correct.mean().mul_(100.0)

                current_lr = self.optim.param_groups[0]["lr"]
                loss_meter.update(loss.item())
                # acc_meter.update(acc.item())
                batch_time.update(time.time() - end)

                # for _c, _y in zip(correct, label):
                #     cls_meters[_y].update(_c.mul_(100.0).item(), n=1)
                # cls_accs = [cls_meters[i].avg for i in range(self.num_classes)]

                # mean_acc = np.mean(np.array(cls_accs))
                # many_acc = np.mean(np.array(cls_accs)[self.many_idxs])
                # med_acc = np.mean(np.array(cls_accs)[self.med_idxs])
                # few_acc = np.mean(np.array(cls_accs)[self.few_idxs])

                meet_freq = (batch_idx + 1) % cfg.print_freq == 0
                only_few_batches = num_batches < cfg.print_freq
                if meet_freq or only_few_batches:
                    nb_remain = 0
                    nb_remain += num_batches - batch_idx - 1
                    nb_remain += (
                        num_epochs - epoch_idx - 1
                    ) * num_batches
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                    info = []
                    info += [f"epoch [{epoch_idx + 1}/{num_epochs}]"]
                    info += [f"batch [{batch_idx + 1}/{num_batches}]"]
                    info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                    info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                    info += [f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]
                    # info += [f"acc {acc_meter.val:.4f} ({acc_meter.avg:.4f})"]
                    # info += [f"(mean {mean_acc:.4f} many {many_acc:.4f} med {med_acc:.4f} few {few_acc:.4f})"]
                    info += [f"lr {current_lr:.4e}"]
                    info += [f"eta {eta}"]
                    print(" ".join(info))

                n_iter = epoch_idx * num_batches + batch_idx
                self._writer.add_scalar("train/lr", current_lr, n_iter)
                self._writer.add_scalar("train/loss.val", loss_meter.val, n_iter)
                self._writer.add_scalar("train/loss.avg", loss_meter.avg, n_iter)
                # self._writer.add_scalar("train/acc.val", acc_meter.val, n_iter)
                # self._writer.add_scalar("train/acc.avg", acc_meter.avg, n_iter)
                # self._writer.add_scalar("train/mean_acc", mean_acc, n_iter)
                # self._writer.add_scalar("train/many_acc", many_acc, n_iter)
                # self._writer.add_scalar("train/med_acc", med_acc, n_iter)
                # self._writer.add_scalar("train/few_acc", few_acc, n_iter)
                
                end = time.time()

            self.sched.step()
            torch.cuda.empty_cache()

        print("Finish training")
        print("Note that the printed training acc is not precise.",
              "To get precise training acc, use option ``test_train True``.")

        # show elapsed time
        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Time elapsed: {elapsed}")

        # save model
        self.save_model(cfg.output_dir)

        self.test()

        # Close writer
        self._writer.close()
    
    @torch.no_grad()
    def test(self, mode="test"):
        if self.tuner is not None:
            self.tuner.eval()
        if self.head is not None:
            self.head.eval()
        self.evaluator.reset()

        if mode == "train":
            print(f"Evaluate on the train set")
            data_loader = self.train_test_loader
        elif mode == "test":
            print(f"Evaluate on the test set")
            data_loader = self.test_loader
        
        write = []
        unlabel = []
        feature_all = []
        label_all = []
        # index = 0
        # t_HC = 0.6

        for batch in tqdm(data_loader, ascii=True):
            image = batch[0]
            label = batch[1]
            # path = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            _bsz, _c, _h, _w = image.size()

            output = self.model(image)
            self.evaluator.process(output, label)


        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"test/{k}"
            if self._writer is not None:
                self._writer.add_scalar(tag, v)
        

        return list(results.values())[0]
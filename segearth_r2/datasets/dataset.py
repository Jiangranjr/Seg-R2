import os
import random
import re
import glob
from dataclasses import dataclass, field
import json
import pathlib

from typing import Dict, Sequence, Optional
import torch
from torch.utils.data import Dataset
import numpy as np
import transformers
import cv2
from PIL import Image
from pycocotools import mask as M

from fvcore.common.config import CfgNode

import warnings

from segearth_r2.utils.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, REFER_TOKEN_INDEX

from segearth_r2.model.mipha import conversation as conversation_lib
from segearth_r2.model import *
from segearth_r2.model.mask_decoder.mask_config.config import Config
from segearth_r2.datasets.augmentation import get_train_augmentation, get_test_augmentation
from segearth_r2.datasets.balanced_sampler import BalancedSampler

warnings.filterwarnings('ignore')
local_rank = None

def preprocess_mask(mask, image_size):
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=0)
    bs, h, w = mask.shape
    processed_masks = []
    for i in range(bs):
        single_mask = mask[i]
        hh, ww = single_mask.shape[:2]
        if ww > hh:
            new_w = image_size
            new_h = int(hh * (image_size / ww))
        else:
            new_h = image_size
            new_w = int(ww * (image_size / hh))
        resized_mask = cv2.resize(single_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        pad_h = image_size - new_h
        pad_w = image_size - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        padded_mask = cv2.copyMakeBorder(resized_mask, top, bottom, left, right, 
                                         cv2.BORDER_CONSTANT, value=0)
        processed_masks.append(padded_mask)
    processed_masks = np.stack(processed_masks, axis=0)
    
    return processed_masks

def preprocess_image(image_path, pad_value=128.0, short_edge_length=1024, max_size=1024, 
                     multiscale_range=None):
    """
    预处理图像，支持多尺度训练
    
    Args:
        image_path: 图像路径
        pad_value: 填充值
        short_edge_length: 短边长度
        max_size: 最大尺寸
        multiscale_range: 多尺度范围，如(0.8, 1.2)，None表示不使用多尺度
    """
    img = Image.open(image_path)
    
    # 多尺度训练：随机缩放短边长度
    if multiscale_range is not None:
        scale_factor = random.uniform(multiscale_range[0], multiscale_range[1])
        short_edge_length = int(short_edge_length * scale_factor)
        max_size = int(max_size * scale_factor)
    
    # ResizeShortestEdge
    w, h = img.size
    size = short_edge_length * 1.0
    scale = size / min(h, w)
    if h < w:
        newh, neww = size, scale * w
    else:
        newh, neww = scale * h, size
    if max(newh, neww) > max_size:
        scale = max_size * 1.0 / max(newh, neww)
        newh = newh * scale
        neww = neww * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    img_resize_shot_edge = img.resize((neww, newh), resample=Image.BILINEAR)
    
    # FixedSizeCrop
    img_np = np.array(img_resize_shot_edge)
    target_size = 1024
    
    # 计算需要的填充或裁剪
    pad_right = max(0, target_size - neww)
    pad_bottom = max(0, target_size - newh)
    
    if pad_right > 0 or pad_bottom > 0:
        # 需要填充
        img_padded = np.pad(
            img_np,
            ((0, pad_bottom), (0, pad_right), (0, 0)),
            mode="constant",
            constant_values=pad_value
        )
    else:
        # 图像已经大于或等于目标尺寸，裁剪中心区域
        if newh > target_size or neww > target_size:
            start_h = (newh - target_size) // 2
            start_w = (neww - target_size) // 2
            img_padded = img_np[start_h:start_h+target_size, start_w:start_w+target_size]
        else:
            img_padded = img_np
    
    return img_padded


def preprocess_image_from_array(img_np, pad_value=128.0, short_edge_length=1024, max_size=1024, multiscale_range=None):
    """从numpy数组预处理图像（用于数据增强后）"""
    # 多尺度训练
    if multiscale_range is not None:
        scale_factor = random.uniform(multiscale_range[0], multiscale_range[1])
        short_edge_length = int(short_edge_length * scale_factor)
        max_size = int(max_size * scale_factor)
    
    h, w = img_np.shape[:2]
    
    # ResizeShortestEdge
    size = short_edge_length * 1.0
    scale = size / min(h, w)
    if h < w:
        newh, neww = int(size), int(scale * w)
    else:
        newh, neww = int(scale * h), int(size)
    if max(newh, neww) > max_size:
        scale = max_size * 1.0 / max(newh, neww)
        newh = int(newh * scale)
        neww = int(neww * scale)
    
    img_resize = cv2.resize(img_np, (neww, newh), interpolation=cv2.INTER_LINEAR)
    
    # FixedSizeCrop - 确保目标尺寸为1024x1024
    target_size = 1024
    pad_right = max(0, target_size - neww)
    pad_bottom = max(0, target_size - newh)
    
    if pad_right > 0 or pad_bottom > 0:
        img_padded = cv2.copyMakeBorder(img_resize, 0, pad_bottom, 0, pad_right, 
                                        cv2.BORDER_CONSTANT, value=pad_value)
    else:
        # 如果图像已经大于目标尺寸，裁剪中心区域
        if neww > target_size or newh > target_size:
            start_h = (newh - target_size) // 2
            start_w = (neww - target_size) // 2
            img_padded = img_resize[start_h:start_h+target_size, start_w:start_w+target_size]
        else:
            img_padded = img_resize
    
    return img_padded

class RS_Base_Dataset(Dataset):
    
    def tokenizer_special_tokens(self, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, refer_token_index=REFER_TOKEN_INDEX, return_tensors=None):
        input_ids = []
        special_token_map = {'<image>': image_token_index, '<refer>':refer_token_index}
        prompt_chunks = re.split('(<image>|<refer>)', prompt)

        for chunk in prompt_chunks:
            if chunk in special_token_map:
                input_ids.append(special_token_map[chunk])
            elif chunk != '':
                input_ids.extend(tokenizer.encode(chunk, add_special_tokens=False))
        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long).squeeze()
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        else:
            return input_ids
        
    def preprocess_llama2(self, sources, tokenizer):
        conv = conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        # Tokenize conversations

        input_ids = torch.stack(
            [self.tokenizer_special_tokens(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)

        targets = input_ids.clone()

        # Mask targets
        sep = conv.sep + conv.roles[1] + ": "
        idx = 0
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)
            if conv.version == 'v0':
                cur_len = 0
                end_token_cnt = 0
                # target[:cur_len] = IGNORE_INDEX
                idx = 0
                for i, rou in enumerate(rounds):
                    if rou == "":
                        continue

                    parts = rou.split(sep)
                    if len(parts) != 2:
                        break
                    parts[0] += sep
                    if idx > 0:
                        round_len = len(self.tokenizer_special_tokens(rou, tokenizer)) + 1
                    else:
                        round_len = len(self.tokenizer_special_tokens(rou, tokenizer)) + 1
                    if idx > 0:
                        instruction_len = len(self.tokenizer_special_tokens(parts[0], tokenizer))
                    else:
                        instruction_len = len(self.tokenizer_special_tokens(parts[0], tokenizer)) - 2

                    target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

                    end_token_cnt += 1
                    cur_len += round_len
                    idx += 1
                target[cur_len:] = IGNORE_INDEX
                cur_len -= end_token_cnt
            else:
                cur_len = 1
                target[:cur_len] = IGNORE_INDEX
                for i, rou in enumerate(rounds):
                    if rou == "":
                        continue

                    parts = rou.split(sep)
                    if len(parts) != 2:
                        break
                    parts[0] += sep
                    round_len = len(self.tokenizer_special_tokens(rou, tokenizer))
                    instruction_len = len(self.tokenizer_special_tokens(parts[0], tokenizer)) - 2

                    target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

                    cur_len += round_len
                    idx += 1
                target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

        return dict(
            input_ids=input_ids,
            labels=targets,
        )

class LaSeRSDataset(RS_Base_Dataset):
    
    def preprocess_referring_instruction(self,instruction, REFER_token='[SEG]'):
        tokenized = self.tokenizer.encode(instruction, add_special_tokens=False)
        REFER_token_id = [self.tokenizer.encode(REFER_token, add_special_tokens=False)[0]]
        tokenized = tokenized + REFER_token_id

        token_refer_id = torch.tensor(tokenized)

        return token_refer_id
    
    def __init__(self, base_data_path, tokenizer, data_args, split='train_data.json', 
                 use_augmentation=False, use_multiscale=False, multiscale_range=(0.8, 1.2),
                 analyze_small_objects=False):
        """
        Args:
            base_data_path: 数据路径
            tokenizer: 分词器
            data_args: 数据参数
            split: 数据集划分
            use_augmentation: 是否使用数据增强
            use_multiscale: 是否使用多尺度训练
            multiscale_range: 多尺度范围，如(0.8, 1.2)
            analyze_small_objects: 是否分析小目标用于后续过采样
        """
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)   
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        
        self.base_data_path = base_data_path
        self.tokenizer = tokenizer
        self.use_augmentation = use_augmentation
        self.use_multiscale = use_multiscale
        self.multiscale_range = multiscale_range
        self.analyze_small_objects = analyze_small_objects
        
        if "train" in split:
            self.LaSeRS_image_path = os.path.join(base_data_path, "train/images")
            self.LaSeRS_json_path = os.path.join(base_data_path, "train/annotations", split)
            # 训练时根据配置决定是否启用数据增强
            if self.use_augmentation:
                self.augmentation = get_train_augmentation()
                print("[Dataset] 训练数据增强已启用")
            else:
                self.augmentation = None
            if self.use_multiscale:
                print(f"[Dataset] 多尺度训练已启用，范围: {multiscale_range}")
        elif "test" in split:
            self.LaSeRS_image_path = os.path.join(base_data_path, "test/images")
            self.LaSeRS_json_path = os.path.join(base_data_path, "test/annotations", split)
            self.augmentation = None  # 测试时不使用增强

        self.SEG_token_id = self.tokenizer.convert_tokens_to_ids("[SEG]")
        
        with open(self.LaSeRS_json_path, "r") as f:
            data = json.load(f)
        self.reason_file = data
        
        # 分析小目标（用于过采样）
        self.mask_areas = None
        if self.analyze_small_objects:
            self.mask_areas = self._analyze_mask_areas()
    
    def _analyze_mask_areas(self):
        """分析所有样本的mask面积，用于小目标检测"""
        print("[Dataset] 分析小目标面积...")
        areas = []
        for data_info in self.reason_file:
            if "mask" in data_info:
                rle_list = data_info['mask']
                total_area = 0
                for rle in rle_list:
                    mask = M.decode(rle)
                    total_area += mask.sum()
                areas.append(total_area)
            else:
                areas.append(0)
        
        areas = np.array(areas)
        small_threshold = 1000  # 小目标阈值
        num_small = (areas < small_threshold).sum()
        print(f"[Dataset] 小目标样本 (<{small_threshold}px): {num_small}/{len(areas)} ({num_small/len(areas)*100:.1f}%)")
        return areas
    
    def get_mask_areas(self):
        """返回mask面积列表，用于过采样器"""
        return self.mask_areas
    
    def __len__(self):
        return len(self.reason_file)
    
    def __getitem__(self, idx):
        data_info = self.reason_file[idx]
        image_path = os.path.join(self.LaSeRS_image_path, data_info['image_name'])
        ref = data_info['description']
        answer = data_info['answer']
        data_id = data_info['id']

        if "mask" in data_info:        
            rle_list = data_info['mask']
            masks = []
            for rle in rle_list:
                mask = M.decode(rle)
                masks.append(mask)
            masks = np.stack(masks, axis=0)
        else:
            masks = None
        
        data_dict = {}
        data_dict['file_name'] = image_path
        image_BGR = cv2.imread(image_path)
        image_height = image_BGR.shape[0]
        image_width = image_BGR.shape[1]
        data_dict['height'] = image_height
        data_dict['width'] = image_width
        data_dict['image_id'] = idx
        
        # ========================================
        # 应用数据增强（仅在训练时且启用增强时）
        # ========================================
        if self.augmentation is not None and masks is not None:
            # BGR转RGB
            image_RGB_raw = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
            # 应用数据增强
            # masks 形状: (num_masks, H, W)，augmentation模块已支持处理
            image_aug, mask_aug = self.augmentation(image_RGB_raw, masks)
            # 转换回BGR用于后续处理
            image_BGR = cv2.cvtColor(image_aug, cv2.COLOR_RGB2BGR)
            # 恢复mask格式为列表
            if mask_aug.ndim == 2:
                # 单个mask
                masks = [mask_aug]
            else:
                # 多个mask: (N, H, W) -> list of (H, W)
                masks = [mask_aug[i] for i in range(mask_aug.shape[0])]
        
        # process image
        # ResizeShortestEdge + FixedSizeCrop
        # 注意：augmentation后的图像直接处理，不需要再读文件
        
        # 确定是否使用多尺度
        multiscale_range = self.multiscale_range if self.use_multiscale else None
        
        if self.augmentation is not None:
            # 如果有增强，直接使用增强后的BGR图像进行预处理
            image_RGB = preprocess_image_from_array(image_BGR, multiscale_range=multiscale_range)
        else:
            # 多尺度训练：传递 multiscale_range 参数
            image_RGB = preprocess_image(image_path, multiscale_range=multiscale_range)
        image_tensor = torch.as_tensor(np.ascontiguousarray(image_RGB.transpose(2, 0, 1)))
        data_dict['image'] = (image_tensor - self.pixel_mean) / self.pixel_std
        
        data_dict['annotations'] = []
        
        mask_num = answer.count("[SEG]")

        for i in range(mask_num):
            data_dict['annotations'].append({
                'data_id': data_id,
                'mask_id': i,
                'mask': np.expand_dims(masks[i], axis=0) if masks is not None else None,
                'image_path': image_path,
                'height': image_height,
                'width': image_width,
                'image_id': os.path.basename(image_path).split(".")[0],
            })
            
        prefix_inst = 'This is an image <|vision_bos|> <image> <|vision_eos|> <|sep|> <|user|>, please doing Reasoning Segmentation according to the following instruction:'
        instruction = ref.strip()
        
        token_refer_id = self.preprocess_referring_instruction(instruction)
        
        sources = [[{'from': 'human', 'value': prefix_inst + '\n<refer> <|assistant|>'},
                    {'from': 'gpt', 'value': '\n' + answer}]]

        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        
        SEG_token_embedding_indices = torch.zeros_like(input_ids)
        SEG_token_embedding_indices[input_ids == self.SEG_token_id] = 1
        
        refer_embedding_indices = torch.zeros_like(input_ids)
        refer_embedding_indices[input_ids == REFER_TOKEN_INDEX] = 1
        
        data_dict['input_ids'] = text_dict['input_ids'][0]
        data_dict['labels'] = text_dict['labels'][0]
        data_dict['dataset_type'] = 'rs_reason_seg'
        
        data_dict['token_refer_id'] = token_refer_id    
        data_dict['refer_embedding_indices'] = refer_embedding_indices
        data_dict['SEG_token_embedding_indices'] = SEG_token_embedding_indices
        
        data_dict['mask_num'] = mask_num
        
        return data_dict

@dataclass
class DataCollatorForCOCODatasetV2(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    clip_image_processor: transformers.SiglipImageProcessor

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        if isinstance(input_ids[0], list):
            BS = len(input_ids)
            T = len(input_ids[0])

            total_input_ids = [k2 for k1 in input_ids for k2 in k1]
            total_input_ids = torch.nn.utils.rnn.pad_sequence(total_input_ids,
                    batch_first=True, padding_value=self.tokenizer.pad_token_id)
            total_input_ids = total_input_ids[:, :self.tokenizer.model_max_length]
            total_labels = [k2 for k1 in labels for k2 in k1]
            total_labels = torch.nn.utils.rnn.pad_sequence(total_labels,
                    batch_first=True, padding_value=self.tokenizer.pad_token_id)
            total_labels = total_labels[:, :self.tokenizer.model_max_length]
            input_ids_batch = []
            labels_batch = []
            for bs in range(BS):
                input_ids_batch.append(total_input_ids[bs*T:(bs+1)*T])
                labels_batch.append(total_labels[bs*T:(bs+1)*T])
                
            input_ids = torch.stack(input_ids_batch, dim=0)
            labels = torch.stack(labels_batch, dim=0)
        else:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                    batch_first=True,
                                                    padding_value=IGNORE_INDEX)
            input_ids = input_ids[:, :self.tokenizer.model_max_length]
            labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            # for video data(key frame, ref frame)
            if isinstance(instances[0]['file_name'], list):
                batch['images_clip'] = []
                for instance in instances:
                    images_file_name = instance['file_name']
                    image_clip = [cv2.imread(image_path) for image_path in images_file_name]
                    image_clip = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in image_clip]
                    image_clip = [self.clip_image_processor.preprocess(
                        img_clip, return_tensors="pt")["pixel_values"][0] for img_clip in image_clip]
                    image_clip = torch.stack(image_clip, dim=0)
                    batch['images_clip'].append(image_clip)
                # bs T c h w
                batch['images_clip'] = torch.stack(batch['images_clip'], dim=0)
            else:
                images_file_name = [instance['file_name'] for instance in instances]
                image_clip = [cv2.imread(image_path) for image_path in images_file_name]
                image_clip = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in image_clip]
                image_clip = [self.clip_image_processor.preprocess(
                    img_clip, return_tensors="pt")["pixel_values"][0] for img_clip in image_clip] #为啥这边输入是1176?
                batch['images_clip'] = torch.stack(image_clip)
            
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        
        for instance in instances:
            for key in ['input_ids', 'labels', 'image']:
                del instance[key]
        if 'instances' in instances[0]:
            batch['seg_info'] = [instance for instance in instances]
        else:
            batch['seg_info'] = []
            for instance_list in instances:
                for seg in instance_list['annotations']:
                    seg['mask'] = torch.as_tensor(seg['mask'], dtype=torch.uint8) if seg['mask'] is not None else None
                    batch['seg_info'].append(seg)
        
        if 'dataset_type' in instances[0]:
            batch['dataset_type'] = [instance['dataset_type'] for instance in instances] # 这个实际上就是那个query

        if 'token_refer_id' in instances[0]:
            token_refer_id = [instance['token_refer_id'] for instance in instances]
            batch['token_refer_id'] = token_refer_id        
        
        if 'SEG_token_embedding_indices' in instances[0]:
            SEG_token_embedding_indices = [instance['SEG_token_embedding_indices'] for instance in instances]
            SEG_token_embedding_indices = torch.nn.utils.rnn.pad_sequence(
                SEG_token_embedding_indices,
                batch_first=True,
                padding_value=0)
            batch['SEG_token_embedding_indices'] = SEG_token_embedding_indices
        
        if 'mask_num' in instances[0]:
            batch['mask_num'] = [instance['mask_num'] for instance in instances]
        
        return batch

class UnifyDatasetSingleDatasetForBatch(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r


    def __init__(self,datasets,dataset_ratio,bs,fix_dataset_len=0):
        super(UnifyDatasetSingleDatasetForBatch, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.fix_dataset_len = fix_dataset_len

        self.cnt = 0
        self.bs = bs

        self.datasets = list(datasets)
        self.datasets_index_list = list(range(len(datasets)))
        self.dataset_ratio = dataset_ratio
        self.cur_dataset_index=0
        self.dataset_length = [len(data) for data in self.datasets]
        self.cumulative_sizes = self.cumsum(self.datasets)
        
    def update_dataset_index(self):
        tempt = self.cur_dataset_index
        tempt += 1
        tempt = tempt % len(self.datasets)
        self.cur_dataset_index = tempt

    def __len__(self):
        if self.fix_dataset_len == 0:
            return self.cumulative_sizes[-1]
        else:
            return self.fix_dataset_len


    def __getitem__(self, idx):
        cur_dataset_len = self.dataset_length[self.cur_dataset_index]
        data_idx = idx % cur_dataset_len
        output_data = self.datasets[self.cur_dataset_index][data_idx]
        self.cnt += 1
        if self.cnt == self.bs:
            self.cnt = 0
            self.update_dataset_index()
        return output_data

def get_mask_config(config='./segearth_r2/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml'):
    cfg_coco = Config.fromfile(config)
    cfg_base = CfgNode.load_yaml_with_base(config, allow_unsafe=True)
    cfg_base.update(cfg_coco.__dict__.items())
    cfg = cfg_base
    cfg = Config(cfg)
    return cfg
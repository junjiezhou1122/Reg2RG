import os
import glob
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import monai.transforms as transforms
from monai.data import PersistentDataset
import nibabel as nib
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from functools import partial
import torch.nn.functional as F
import tqdm
import random
import pickle
import hashlib
from typing import Any, Dict, List, Tuple

CONDITIONS = [
    'Medical material',
    'Arterial wall calcification',
    'Cardiomegaly',
    'Pericardial effusion',
    'Coronary artery wall calcification',
    'Hiatal hernia',
    'Lymphadenopathy',
    'Emphysema',
    'Atelectasis',
    'Lung nodule',
    'Lung opacity',
    'Pulmonary fibrotic sequela',
    'Pleural effusion',
    'Mosaic attenuation pattern',
    'Peribronchial thickening',
    'Consolidation',
    'Bronchiectasis',
    'Interlobular septal thickening',
]

REGIONS = [
    'abdomen',
    'bone',
    'breast',
    'esophagus',
    'heart',
    'lung',
    'mediastinum',
    'pleura',
    'thyroid',
    'trachea and bronchie',
]


class RadGenomePersistentCacheTransform:
    """
    Deterministic preprocessing for MONAI PersistentDataset caching.

    Caches the expensive part (NIfTI loading + crop/resize) so future epochs
    can reuse cached tensors from disk and reduce CPU/IO pressure.

    The transform returns:
    - img_hu: torch.Tensor, shape (1, H, W, D) in HU space (not normalized)
    - seg: torch.Tensor, shape (N, H, W, D) resized masks (float/binary)
    - mask_keys: List[str], region names aligned with seg[0..N-1]
    """

    def __init__(self, target_size: Tuple[int, int, int] = (256, 256, 64)) -> None:
        self.target_size = target_size

        def threshold(x: np.ndarray) -> np.ndarray:
            return x > -1000

        self._image_transform = transforms.Compose(
            [
                transforms.CropForegroundd(
                    keys=["img", "seg"], source_key="img", select_fn=threshold
                ),
                transforms.Resized(
                    keys=["img", "seg"],
                    spatial_size=self.target_size,
                    mode=("trilinear", "nearest"),
                ),
                transforms.ToTensord(keys=["img", "seg"]),
            ]
        )

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        img_path = sample["image"]

        mask_paths: List[str] = []
        mask_keys: List[str] = []
        for region in REGIONS:
            if region in sample:
                mask_paths.append(sample[region])
                mask_keys.append(region)

        img_data = nib.load(img_path).get_fdata().astype(np.float32)
        img_data = img_data[np.newaxis, ...]  # (1, H, W, D)

        masks: List[np.ndarray] = []
        for mask_path in mask_paths:
            masks.append(nib.load(mask_path).get_fdata().astype(np.float32))
        seg_data = np.stack(masks, axis=0)  # (N, H, W, D)

        tensors = self._image_transform({"img": img_data, "seg": seg_data})
        return {
            "img_hu": tensors["img"],
            "seg": tensors["seg"],
            "mask_keys": mask_keys,
        }
    
class RadGenomeDataset_Train(PersistentDataset):
    def __init__(self, text_tokenizer, data_folder, mask_folder, csv_file, cache_dir, max_region_size=10, max_img_size = 1, image_num = 32, region_num=33, max_seq=2048, resize_dim=500, voc_size=32000, force_num_frames=True):
        # text_tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained(
                text_tokenizer,
        )
        
        # NOTE: "additoinal_special_tokens" is a type of special token used in tokenizer
        special_token = {
            "additional_special_tokens": ["<image>", "</image>", "<region>", "</region>"]}
        # 'max_img_size' is the max number of images in a single input,
        # 'image_num' is the max number of image tokens in a single image
        self.image_padding_tokens = []
        for i in range(max_img_size):
            image_padding_token = ""
            for j in range(image_num):
                image_token = "<image"+str(i*image_num+j)+">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append(
                    "<image"+str(i*image_num+j)+">")
            self.image_padding_tokens.append(image_padding_token)
        
        self.region_padding_tokens = []
        for i in range(max_region_size):
            region_padding_tokens = ""
            for j in range(region_num):
                region_token = "<region"+str(i*region_num+j)+">"
                region_padding_tokens = region_padding_tokens + region_token
                special_token["additional_special_tokens"].append(
                    "<region"+str(i*region_num+j)+">")
            self.region_padding_tokens.append(region_padding_tokens)

        self.text_tokenizer.add_special_tokens(
            special_token
        )
        # treat the token with ID 0 as the pad token
        self.text_tokenizer.pad_token_id = 0
        # treat the token with ID 1 as the bos token
        self.text_tokenizer.bos_token_id = 1
        # treat the token with ID 2 as the eos token
        self.text_tokenizer.eos_token_id = 2
        
        self.voc_size = voc_size
        self.max_seq = max_seq
        self.data_folder = data_folder
        self.mask_folder = mask_folder
        self.csv_file = csv_file

        self.accession_to_sentences = self.load_accession_sentences(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.target_size = (256, 256, 64) #NOTE: the target input size of the image
        self._cache_transform = RadGenomePersistentCacheTransform(target_size=self.target_size)
        self._region_resize = transforms.Resize(spatial_size=self.target_size, mode="trilinear")

        super().__init__(data=self.samples, transform=self._cache_transform, cache_dir=cache_dir)

    def load_accession_sentences(self, xlsx_file):
        df = pd.read_csv(xlsx_file)
        df_grouped = df.groupby('Volumename')

        accession_to_sentences = {}
        for accession, group in df_grouped:
            # 将后缀为.nii.gz替换为.npz
            # accession = accession.replace('.nii.gz', '.npz')
            sentences = {}
            for i, row in group.iterrows():
                if pd.isna(row['Anatomy']):
                    anatomy_key = 'whole'
                else:
                    anatomy_key = row['Anatomy']
                sentences[anatomy_key] = row['Sentence']
            accession_to_sentences[accession] = sentences
        return accession_to_sentences

    def prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))
        
        # 获取当前文件的绝对路径
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        cache_file = self._get_samples_cache_path(current_file_dir)

        if os.path.exists(cache_file):
            samples = pickle.load(open(cache_file, 'rb'))
        else:
            for patient_folder in tqdm.tqdm(patient_folders):
                accession_folders = glob.glob(os.path.join(patient_folder, '*'))

                for accession_folder in accession_folders:
                    nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz'))
                    for nii_file in nii_files:
                        accession_number = os.path.basename(nii_file)

                        if accession_number not in self.accession_to_sentences:
                            continue
                            
                        single_sample = {}
                        single_sample["image"] = nii_file
                        single_sample["accession"] = accession_number

                        volume_name = accession_number.split(".")[0]
                        mask_path = os.path.join(self.mask_folder, 'seg_'+volume_name)

                        flag = False
                        for region in REGIONS:
                            # NOTE: only use the samples with the corresponding region report
                            if region in self.accession_to_sentences[accession_number]:
                                mask_file = os.path.join(mask_path, region + '.nii.gz')
                                if not os.path.exists(mask_file):
                                    continue
                                single_sample[region] = mask_file
                                flag = True
                        if not flag: # NOTE: if there is no corresponding region report, skip this sample
                            continue

                        samples.append(single_sample)
                        self.paths.append(nii_file)

            # save the samples to a file
            with open(cache_file, 'wb') as f:
                pickle.dump(samples, f)

        print('Number of samples: ', len(samples))

        return samples

    def _get_samples_cache_path(self, current_file_dir: str) -> str:
        """
        Cache sample list per (data_folder, mask_folder, csv_file) to avoid
        train/val collisions when multiple datasets are instantiated.
        """
        key = f"{self.data_folder}|{self.mask_folder}|{self.csv_file}"
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        return os.path.join(current_file_dir, f"samples_{digest}.pkl")

    def __len__(self):
        return len(self.samples)

    def _mask_bbox(self, mask: torch.Tensor) -> Tuple[int, int, int, int, int, int]:
        """
        Compute (h0, h1, w0, w1, d0, d1) bbox for a 3D mask tensor (H, W, D).
        Assumes mask contains at least one positive voxel.
        """
        idx = mask.nonzero(as_tuple=False)
        mins = idx.min(dim=0).values
        maxs = idx.max(dim=0).values
        h0, w0, d0 = (int(v) for v in mins.tolist())
        h1, w1, d1 = (int(v) for v in maxs.tolist())
        return h0, h1, w0, w1, d0, d1

    def text_add_image_tokens(self, text):
        
        text = '<image>' + self.image_padding_tokens[0] + '</image>' + '. ' + text
        text = "The global information is provided as the context: " + text

        return text

    def text_add_region_tokens(self, text, num_regions):
        region_text = ""
        for i in range(num_regions):
            region_text = region_text + "The region " + str(i) + " is " + '<region>' + self.region_padding_tokens[i] + '</region>. ' 
        text = region_text + text

        return text

    def __getitem__(self, index):
        raw = self.data[index]
        img_file = raw["image"]
        accession = raw.get("accession", os.path.basename(img_file))

        region_reports = {}
        mask_files = {}
        for key, value in raw.items():
            if key in {"image", "accession"}:
                continue
            mask_files[key] = value
            region_reports[key] = self.accession_to_sentences[accession][key]

        cached = super().__getitem__(index)
        img_hu: torch.Tensor = cached["img_hu"]  # (1, H, W, D)
        seg: torch.Tensor = cached["seg"]        # (N, H, W, D)
        mask_keys: List[str] = cached["mask_keys"]

        hu_min, hu_max = -1000, 200
        global_img = torch.clamp(img_hu, hu_min, hu_max)
        global_img = ((global_img + 400) / 600).float()
        global_img = global_img.repeat(3, 1, 1, 1).unsqueeze(0)  # (1, 3, H, W, D)

        mask_img_tensors: Dict[str, torch.Tensor] = {"image": global_img}
        mask_tensors: Dict[str, torch.Tensor] = {}

        # Build region-specific tensors from cached (img_hu, seg) to avoid re-loading NIfTI.
        # We threshold seg to a binary mask because resizing may introduce interpolation.
        for i, key in enumerate(mask_keys):
            if key not in region_reports:
                continue
            mask = (seg[i] > 0.5)
            if mask.sum().item() == 0:
                continue

            h0, h1, w0, w1, d0, d1 = self._mask_bbox(mask)
            img_crop = img_hu[:, h0 : h1 + 1, w0 : w1 + 1, d0 : d1 + 1].clone()
            mask_crop = mask[h0 : h1 + 1, w0 : w1 + 1, d0 : d1 + 1]

            img_crop[:, ~mask_crop] = -1024
            img_crop = self._region_resize(img_crop)  # (1, H, W, D) -> target_size

            img_crop = torch.clamp(img_crop, hu_min, hu_max)
            img_crop = ((img_crop + 400) / 600).float()
            img_crop = img_crop.repeat(3, 1, 1, 1).unsqueeze(0)  # (1, 3, H, W, D)

            mask_img_tensors[key] = img_crop
            mask_tensors[key] = seg[i].unsqueeze(0)

        #NOTE: remove useless regions from region_reports according to mask_img_tensors
        for key in list(region_reports.keys()):
            if key not in mask_img_tensors:
                region_reports.pop(key)

        region2area = {}
        shuffled_areas = list(region_reports.keys())
        random.shuffle(shuffled_areas)

        for i in range(len(shuffled_areas)):
            region2area[i] = shuffled_areas[i]

        instruction = "Given the provided global and regional information from this CT scan, please generate a comprehensive medical report for each region. First, identify the anatomical area corresponding to each region, then provide detailed information about these anatomical structures and any abnormalities that are essential. You can refer to the global information as the context and take it as a supplement when generating each region report."
        
        # prompt = self.text_add_image_tokens(instruction)
        prompt = self.text_add_region_tokens(instruction, num_regions=len(region2area))

        prompt = self.text_add_image_tokens(prompt)

        combined_report = ""
        for i in range(len(region2area)):
            area = region2area[i]
            region_report = region_reports[area]
            combined_report = combined_report + "The region " + str(i) + " is " + area + ": " + region_report + " "

        # make text input
        self.text_tokenizer.padding_side = "right"
            
        text_tensor = self.text_tokenizer(
            prompt + ' ' + combined_report, max_length=self.max_seq, truncation=True, padding="max_length", return_tensors="pt"
        )
        text_input = text_tensor["input_ids"][0]
        attention_mask = text_tensor["attention_mask"][0]

        text_input[torch.sum(attention_mask)] = self.text_tokenizer.eos_token_id  # set the last token id to be eos id

        # make label
        prompt_tensor = self.text_tokenizer(
            prompt, max_length=self.max_seq, truncation=True, padding="max_length", return_tensors="pt"
        )
        prompt_length = torch.sum(prompt_tensor["attention_mask"][0])

        label = text_input.clone()
        label[label == self.text_tokenizer.pad_token_id] = -100
        # remove the additional special tokens from the label
        label[label >= self.voc_size] = -100
        label[:prompt_length] = -100  # only focus on answer part

        return {'lang_x': text_input, 
                'vision_x': mask_img_tensors, 
                'mask_x': mask_tensors,
                'region2area': region2area,
                'attention_mask': attention_mask, 
                'label': label
                }

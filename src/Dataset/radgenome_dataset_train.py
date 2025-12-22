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
from monai.transforms import Transform
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


class RadGenomePersistentCacheTransform(Transform):
    """
    MONAI-compliant deterministic transform for PersistentDataset caching.

    Inherits from Transform so MONAI recognizes it as cacheable (non-randomizable).
    Loads NIfTI files, applies crop/resize, and adds processed tensors to the data dict.

    MONAI PersistentDataset will cache the output of this transform automatically.
    """

    def __init__(self, target_size: Tuple[int, int, int] = (256, 256, 64)) -> None:
        super().__init__()
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

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform that loads and processes NIfTI files.

        CRITICAL: Must return the SAME dict object (modified in-place) or a new dict
        that contains ALL original keys plus new processed keys. This allows MONAI
        to cache the full result properly.
        """
        # Make a copy to avoid modifying the original during caching
        d = dict(data)

        img_path = d["image"]

        # Collect mask paths and keys
        mask_paths: List[str] = []
        mask_keys: List[str] = []
        for region in REGIONS:
            if region in d:
                mask_paths.append(d[region])
                mask_keys.append(region)

        # Load image and masks from NIfTI files
        img_data = nib.load(img_path).get_fdata().astype(np.float32)
        img_data = img_data[np.newaxis, ...]  # (1, H, W, D)

        masks: List[np.ndarray] = []
        for mask_path in mask_paths:
            masks.append(nib.load(mask_path).get_fdata().astype(np.float32))
        seg_data = np.stack(masks, axis=0)  # (N, H, W, D)

        # Apply MONAI transforms (crop, resize, to tensor)
        tensors = self._image_transform({"img": img_data, "seg": seg_data})

        # Add processed tensors to the dict (MONAI will cache this full dict)
        d["img_hu"] = tensors["img"]
        d["seg"] = tensors["seg"]
        d["mask_keys"] = mask_keys

        return d

class RobustPersistentDataset(PersistentDataset):
    """
    PersistentDataset with atomic writes to prevent cache corruption on interruption.
    
    Key improvements:
    - Atomic writes: tmp file → rename (interrupted writes don't corrupt cache)
    - Skip corrupted files: if a file is corrupted, recompute instead of crashing
    - Safe for interruption and multi-user environments
    """
    
    def _cachecheck(self, item_transformed):
        """
        Override MONAI's _cachecheck to add atomic write and error handling.
        """
        hashfile = self._get_hashfile(item_transformed)
        
        # Try to load existing cache
        if os.path.exists(hashfile):
            try:
                return torch.load(hashfile, map_location="cpu")
            except Exception as e:
                # Corrupted cache file - delete and recompute
                print(f"Warning: Corrupted cache {hashfile}, recomputing... ({e})")
                try:
                    os.remove(hashfile)
                except:
                    pass
        
        # Cache miss or corrupted - compute with transform
        result = self.transform(item_transformed)
        
        # Atomic write: write to .tmp then rename (crash-safe)
        tmp_file = hashfile + ".tmp"
        try:
            torch.save(result, tmp_file)
            os.replace(tmp_file, hashfile)  # Atomic on POSIX systems
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
            if os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except:
                    pass
        
        return result
    
    def _get_hashfile(self, item):
        """Get the cache filename for an item (uses MONAI's hashing)."""
        # Use MONAI's internal hashing function
        hash_val = self.hash_func(item)

        # Fix: Ensure hash_val is a proper string (not bytes with b'...' literal)
        if isinstance(hash_val, bytes):
            hash_val = hash_val.hex()  # Convert bytes to hex string
        elif not isinstance(hash_val, str):
            hash_val = str(hash_val)  # Fallback for other types

        # Add transform hash if available
        if hasattr(self, 'hash_transform') and self.hash_transform is not None:
            transform_hash = self.hash_transform(self.transform)
            # Ensure transform hash is also a string
            if isinstance(transform_hash, bytes):
                transform_hash = transform_hash.hex()
            hash_val += str(transform_hash)

        return os.path.join(self.cache_dir, f"{hash_val}.pt")
    
    def cache_exists(self, index: int, deep_check: bool = False) -> bool:
        """
        Fast check if cache exists and is valid for a given index.
        Used for efficient warmup progress tracking.

        Args:
            index: Sample index to check
            deep_check: If True, actually load the file to verify integrity (slower but safer)

        Returns True only if:
        1. Cache file exists (.pt)
        2. File is not empty (basic sanity check)
        3. (Optional) File can be loaded without corruption
        """
        item = self.data[index]
        hashfile = self._get_hashfile(item)

        # Quick check: file must exist and have non-zero size
        if not os.path.exists(hashfile):
            return False

        # Basic integrity check: file should not be empty
        try:
            file_size = os.path.getsize(hashfile)
            if file_size == 0:
                # Empty file - corrupted, remove it
                os.remove(hashfile)
                return False
        except (OSError, IOError):
            return False

        # Optional: Deep validation (loads the file to verify integrity)
        if deep_check:
            try:
                torch.load(hashfile, map_location="cpu")
                return True
            except Exception:
                # Corrupted - delete and return False
                print(f"Warning: Corrupted cache detected at {hashfile}, will recompute")
                try:
                    os.remove(hashfile)
                except:
                    pass
                return False

        return True


class RadGenomeDataset_Train(RobustPersistentDataset):
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
        # Bump this string when the cached transform/output schema changes.
        # It becomes part of each sample dict and therefore part of MONAI's cache key.
        self.cache_version = "lit_img_hu_seg_v1"

        self.accession_to_sentences = self.load_accession_sentences(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.target_size = (256, 256, 64) #NOTE: the target input size of the image
        self._cache_transform = RadGenomePersistentCacheTransform(target_size=self.target_size)
        self._region_resize = transforms.Resize(spatial_size=self.target_size, mode="trilinear")
        
        # Initialize PersistentDataset with samples and transform
        # MONAI will automatically cache the transform output to cache_dir
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
            # Forward-compat: ensure required fields exist in cached sample lists.
            for sample in samples:
                if isinstance(sample, dict):
                    sample.setdefault("_cache_version", self.cache_version)
                    # Older cached sample lists may store [mask_path, report] tuples/lists.
                    for region in REGIONS:
                        value = sample.get(region)
                        if isinstance(value, (list, tuple)) and value:
                            sample[region] = value[0]
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
                        single_sample["_cache_version"] = self.cache_version

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
        key = f"{self.data_folder}|{self.mask_folder}|{self.csv_file}|{self.cache_version}"
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
        # Get cached data from PersistentDataset (includes original keys + processed tensors)
        # MONAI has already run the transform and cached the result
        cached = super().__getitem__(index)

        # Extract metadata from cached dict
        img_file = cached["image"]
        accession = cached.get("accession", os.path.basename(img_file))

        # Build region reports from cached metadata
        region_reports = {}
        for key in REGIONS:
            if key in cached and key in self.accession_to_sentences.get(accession, {}):
                region_reports[key] = self.accession_to_sentences[accession][key]
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

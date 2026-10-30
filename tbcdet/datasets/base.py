import os.path as osp

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self,
                 ann_file,
                 classes,
                 data_root = None):
        self.data_root = data_root
        self.ann_file = ann_file if data_root is None else osp.join(data_root, ann_file)
        
        self.classes = classes
        self.data_infos = self.load_annotations(self.ann_file)
        
    def __len__(self):
        return len(self.data_infos)
    
    def __getitem__(self, idx):
        return self._prepare_img(idx)
    
    def load_annotations(self, ann_file):
        return ...
    
    def _prepare_img(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        
        results = dict(img_info=img_info, ann_info=ann_info)
        
        self.pre_pipeline(results)
        return self.pipeline(results)
    
    def get_ann_info(self, idx):
        ...

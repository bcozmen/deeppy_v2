from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import glob

class GraphNeRFDataset(Dataset):
    def __init__(self, data_path, load_spatial=False, nerf_types = ["mlp", "hash" ,"triplane"], label_types= [0]):
        self.data_path = data_path
        self.all_classes, self.all_objects = self._gather_all_classes()
        self.nerf_types = nerf_types
        self.load_spatial = load_spatial
        if type(label_types) is int:
            label_types = [label_types]
        self.label_types = label_types


    def __len__(self):
        return len(self.all_objects)

    def __getitem__(self, idx):
        obj = self.all_objects[idx]
        class_name, obj_type = obj.split("/")

        
        
        item = []
        for ntype in self.nerf_types:
            V, E = self._load_object(obj, ntype=ntype)
            item.extend([V, E])

        labels = []
        for ntype in self.nerf_types:
            histograms = self._load_labels(obj, ntype=ntype)
            labels.append(histograms)
        
        labels = torch.stack(labels, dim=0)[:, self.label_types]  # (num_nerf_types, 6, len(label_types))
        item.append(labels)
        
        return item

    def _load_object(self,obj_name, ntype = "mlp"):
        weights = torch.load(f"{self.data_path}/{ntype}/{obj_name}/nerf_graph_weights.pth")

        v,e = weights["v"], weights["e"]
        if not self.load_spatial and ntype == "triplane":
            n = 3*32*32
            ne = 3*32*32*16

            v = v[n:]
            e = e[ne:]
            
            e[...,0] -= n
            e[...,1] -= n

        if not self.load_spatial and ntype == "hash":
            n = 4096 * 4
            ne = 4096 * 4 * 2
            
            v = v[n:]
            e = e[ne:]
            
            e[...,0] -= n
            e[...,1] -= n
        return v, e
    
    def _load_labels(self,obj_name, ntype = "mlp"):
        histograms = torch.load(f"{self.data_path}/{ntype}/{obj_name}/histograms.pth") # (6,10)
        return histograms["histograms"]

    def _gather_all_classes(self):
        all_classes = glob.glob(f"{self.data_path}/*/*/*/nerf_graph_weights.pth")
        all_classes = list(set([obj.split("/")[-3] for obj in all_classes]))
        all_classes.sort()
        all_objects = []

        pos_weights = torch.load(f"{self.data_path}/pos_weights.pth")
        self.volume_indices = pos_weights['volume_indices']
        
        for obj in all_classes:
            all_vars = [obj + "/" + k.split("/")[-2] for k in glob.glob(f"{self.data_path}/*/{obj}/*/nerf_graph_weights.pth")]
            all_objects.extend(list(set(all_vars)))
        return all_classes, all_objects


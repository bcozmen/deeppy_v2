from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import glob

class GraphNeRFDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.all_classes, self.all_objects = self._gather_all_classes()
        self.nerf_types = ["mlp" , "hash", "triplane"]


    def __len__(self):
        return len(self.all_objects)

    def __getitem__(self, idx):
        obj = self.all_objects[idx]
        class_name, obj_type = obj.split("/")

        label = torch.tensor([self.all_classes.index(class_name), idx, 0], dtype=torch.long)
        
        item = []
        for ntype in self.nerf_types:
            V, E = self._load_object(obj, ntype=ntype)
            item.extend([V, E])
        item.append(label)

        return item

    def _load_object(self,obj_name, ntype = "mlp"):
        weights = torch.load(f"{self.data_path}/{ntype}/{obj_name}/nerf_graph_weights.pth")
        V = weights["v"]
        E = weights["e"]
        if ntype == "mlp":
            V = V.unsqueeze(-1)
        return V,E




    def _gather_all_classes(self):
        all_classes = glob.glob(f"{self.data_path}/*/*/*/nerf_weights.pth")
        all_classes = list(set([obj.split("/")[-3] for obj in all_classes]))
        all_classes.sort()
        all_objects = []
        
        for obj in all_classes:
            all_vars = [obj + "/" + k.split("/")[-2] for k in glob.glob(f"{self.data_path}/*/{obj}/*/nerf_weights.pth")]
            all_objects.extend(list(set(all_vars)))
        return all_classes, all_objects


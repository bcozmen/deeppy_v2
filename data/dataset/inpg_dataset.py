from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import glob

import numpy as np
import pickle


class GridEncoder():
    def __init__(self, levels = 16, base_resolution = 16,  align_corners = True, hashmap_size = 2**19):
        super(GridEncoder, self).__init__()
        self.base_resolution = torch.tensor(base_resolution)
        self.align_corners = align_corners
        
        self.num_levels = levels
        self.levels = torch.arange(levels)
        
        self.per_level_scale = torch.exp2(torch.log2(2048 / self.base_resolution) / (self.num_levels - 1))
        
        
        
        #(16,1)
        self.resolutions = torch.ceil(self.base_resolution * (self.per_level_scale ** self.levels)).to(torch.int64).unsqueeze(1)
        
        self.hashmap_sizes = torch.min(self.resolutions ** 3, torch.tensor(hashmap_size)).view(1,1, self.levels[-1]+1,1)

        self.hash_table_indices_end = torch.cumsum(self.hashmap_sizes.flatten(), dim=0).to(dtype=torch.int64)
        self.hash_table_indices_start = torch.cat([torch.tensor([0]), self.hash_table_indices_end[:-1]])
        #(1,16,1)
        scale = torch.exp2(self.levels * torch.log2(self.per_level_scale)) * self.base_resolution - 1.0
        self.scale = scale.unsqueeze(1).unsqueeze(0)
        
        # Get 8 offset combinations: all binary 3D offsets (0 or 1 for each axis)
        #(8,3)
        self.offsets = torch.tensor([[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]], dtype=torch.int64)  
        
        # Reshape for broadcasting: [1, 1, 8, 3]
        self.offsets = self.offsets.view(1, 1, -1,3)
    def __call__(self,xyz):
        return self.forward(xyz)
    def fast_hash(self,xyz_corners, primes=(1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737)):
        primes = torch.tensor(primes[:xyz_corners.shape[-1]]).to(torch.int64).unsqueeze(0)

        xyz_primes = xyz_corners * primes
        return np.bitwise_xor.reduce(xyz_primes, axis=1).unsqueeze(1)
    def forward(self, xyz):
        #Input in Shape (B,D)
        B,D = xyz.shape[0], xyz.shape[1]
        
        level = self.num_levels #num_levels = 16
        #(B,16, D)
        xyz_per_level = xyz.repeat_interleave(level, dim=0).view(B, level, D)
        
        #weights = self.compute_corner_weights(xyz_per_level)
        
        #(B,16,1,3)
        xyz_scaled = torch.floor((xyz_per_level * self.scale) + (0.0 if self.align_corners else 0.5)) # self.scale shape = torch.Size([1, 16, 1])
        xyz_scaled = xyz_scaled.to(torch.int64).unsqueeze(2)
        
        #(512,16,8,3)
        xyz_corners = xyz_scaled + self.offsets # offsets shape = torch.Size([1, 1, 8, 3])
        orig_shape = xyz_corners.shape[:-1]
        
        #(-1,3)
        xyz_corners = xyz_corners.view(-1,3)
        
        #(B*16*8,1)
        hashes = self.fast_hash(xyz_corners)
        
        
        #(B,8,16,1)
        hashes = hashes.view(B,16,8,1)
        hashes = hashes.permute(0,2,1,3)
       
        
        
        indices_layerwise = hashes % self.hashmap_sizes # hashmap_sizes shape torch.Size([1, 1, 16, 1])

        #return (B,8,16,1)
        indices_global = indices_layerwise +  self.hash_table_indices_start.view(1,1,-1,1) #hash_table_indices_start shape torch.Size([16])
 
        return indices_global, indices_layerwise
        
    def compute_corner_weights(self,xyz_per_level):
        """
        Args:
            points: [N, d] float — continuous positions
        Returns:
            weights: [N, 2^d] — interpolation weights for each corner
        """
        xyz_scaled = (xyz_per_level * self.scale) + (0.0 if ge.align_corners else 0.5)
        
        # (512,16,3)
        w = xyz_scaled - torch.floor(xyz_scaled)  # fractional offsets [0,1)
        
        #(512*16,3)
        w = w.reshape(-1,3)
        N = w.shape[0]

        # Expand to [N, 2^d, d]
        w_exp = w.unsqueeze(1).expand(-1, 2 ** 3, -1)              # [N, 2^d, d]

        offsets = ge.offsets.squeeze(0).expand(N, -1, -1)    # [N, 2^d, d]


        # For each dimension: select w_l or (1 - w_l) depending on offset
        weights = torch.where(offsets.bool(), w_exp, 1 - w_exp)    # [N, 2^d, d]

        # Multiply over d dimensions to get final weight per corner
        weights = weights.prod(dim=-1)    
        weights = weights.reshape(512,16,8)
        
        return weights.unsqueeze(-1)
    

class IngpDataset(Dataset):
    def __init__(self, data_path, config, window_size = None, token_size = None, max_layer_width = 64, device = None, train = True, 
                 mlp_token_size = 53, pos_token_size = 10, data_buffer_size = 5,  permutation_augment = True, normalize = False):
        self.data_path = data_path
        self.config = config

        base_resolution = self.config['hash_encoding']["base_resolution"]  # Base resolution of the grid encoder
        num_levels = self.config['hash_encoding']["num_levels"]  # Number of levels in the grid encoder
        self.grid_encoder = GridEncoder(base_resolution=base_resolution, levels=num_levels, hashmap_size= 2**19)

        #self.pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        self.data_buffer_size = data_buffer_size
        self.normalize = normalize
        self.permutation_augment = permutation_augment
        self.window_size = window_size
        self.pos_token_size = pos_token_size
        self.mlp_token_size = mlp_token_size
        self.hash_chunk_size = self.window_size - self.mlp_token_size - self.pos_token_size# 53 is the size of the MLP tokens # And one for rotation index
        self.token_size = token_size
        self.max_layer_width = max_layer_width

        if self.token_size % self.max_layer_width != 0:
            raise ValueError("Invalid token size or max_layer_width")
        
        self.load_object_paths()
        
        #VANILLA
        #self.max_positions = self[0][1].max(axis=0).values + 2
        #hash_table_indices_end = self.grid_encoder.hash_table_indices_end
        #self.max_positions[2] = (hash_table_indices_end[-1]  - hash_table_indices_end[-2] ).item() + 1

        #EXP1
        self.max_positions = torch.tensor([self.grid_encoder.hash_table_indices_end[-1]+1 ,self.mlp_token_size + self.pos_token_size])

    def __len__(self):
        return len(self.all_objects_2d)

    def load_object_paths(self): 
        self.objects = glob.glob(self.data_path + "/*")
        self.all_objects_2d = []
        self.all_objects = []

        try:
            with open('object_lists.pkl', 'rb') as f:
                self.all_objects, self.all_objects_2d = pickle.load(f)
            
        except:
            #For all objects
            for o in self.objects:
                #Get paths for (up to) 6 different views
                obj_paths = glob.glob(o + "/**/hash.bin",  recursive=True)
                #If no augmentation, skip
                if len(obj_paths) <= 1:
                    continue
                #From the folder name calculate the transform
                transforms =  [[float(m) for m in k.split("/")[-3].split("_")[1:]] for k in obj_paths]
                transforms = torch.deg2rad(torch.tensor(transforms, dtype=torch.float32))  # Convert degrees to radians
                
                obj_paths = [o.replace("hash.bin","") for o in obj_paths]
                this_object = list(zip(*[obj_paths, transforms]))
                
                self.all_objects.extend(this_object)
                self.all_objects_2d.append(this_object)

        self.data_buffer = [[] for _ in self.all_objects_2d]
        
        #with open('object_lists.pkl', 'wb') as f:
        #    pickle.dump((self.all_objects, self.all_objects_2d), f)  # Store as a tuple
   

    def __getitem__(self, idx):
        idx = idx
        #Random index
        #Sample (window_size - hash_chunk_size )points in 3D space (512,3)
        #points = torch.rand((self.hash_chunk_size * 4, 3)
        points = torch.rand(self.hash_chunk_size, 3)
        points = points.clamp(0.0,1.0)                
        #points1 = points[torch.randperm(len(points))[:self.hash_chunk_size]]
        #points2 = points[torch.randperm(len(points))[:self.hash_chunk_size]]
        
        
        #Get 2 random views of the object
        object_parent_path = self.all_objects_2d[idx]
        idx_child = torch.randperm(len(object_parent_path))[:2]

        obj1_path, obj_1_transform = object_parent_path[idx_child[0]]
        obj2_path, obj_2_transform = object_parent_path[idx_child[1]] 

        [t1,p1,m1], r1 = self.load_weights(obj1_path, points), obj_1_transform
        (t2,p2,m2), r2 = self.load_weights(obj2_path, points), obj_2_transform
        
        r1 += torch.randn_like(r1) * torch.deg2rad(torch.tensor(1))
        r2 += torch.randn_like(r2) * torch.deg2rad(torch.tensor(1))
        return t1.numpy(), p1.numpy(), m1.numpy(), r1.numpy(), t2.numpy(), p2.numpy(), m2.numpy(), r2.numpy()
        
    def load_weights(self, file_path, points):
        file_path = file_path.replace("final.pth", "")
        # 3D points = (C,3)
        C = points.shape[0]
       
        #(C,8,16,1)
        indices_global, indices_layerwise  = self.grid_encoder(points)
        indices_layers = torch.arange(indices_global.shape[2]).repeat(C,indices_global.shape[1],1).unsqueeze(-1)

       

        indices_flat = indices_global.flatten()
        indices_sorted, original_order = torch.sort(indices_flat, descending=False)
        
        unique_indices, inverse_indices = torch.unique(indices_sorted, return_inverse=True)

        shape = (6537456, 2)        
        mmap_raw = np.memmap(file_path + "hash.bin", dtype=np.float32, mode='r', shape=shape)
        batch_np = mmap_raw[unique_indices.numpy()]
        result = torch.from_numpy(batch_np)
        
        
        hash_tokens_sorted = result[inverse_indices]                  # shape: (num_total, 2)
        hash_tokens = hash_tokens_sorted[original_order.argsort()]  

        #(C,256)
        hash_tokens = hash_tokens.reshape(C,-1)
        
        #Augment
        
        hash_masks = torch.ones_like(hash_tokens)
        #(C,128)


        #hash_pos = indices.reshape(B,-1)

        
        #(6 , X , token_size) -> (1, 6X, token_size)
        data = torch.load(file_path + "mlp_raw.pth")
        geometry_layers, view_layers = self.permute_mlp_weights(data["geometry_layers"]), self.permute_mlp_weights(data["view_layers"])
        
        if self.normalize:
            hash_tokens = torch.nn.functional.normalize(hash_tokens, p=2, dim=-1)
            for ix in range(len(geometry_layers)):
                geometry_layers[ix] = torch.nn.functional.normalize(geometry_layers[ix], p=2, dim=-1)
                view_layers[ix] = torch.nn.functional.normalize(view_layers[ix], p=2, dim=-1)

        for ix in range(len(geometry_layers)):
            geometry_layers[ix] = self.tokenize_mlp_layer(geometry_layers[ix])
            view_layers[ix] = self.tokenize_mlp_layer(view_layers[ix])
        mlp_tokens,mlp_masks = [torch.vstack(k) for k in zip(*(geometry_layers + view_layers))]
        
        if False and self.normalize:
            hash_tokens = torch.nn.functional.normalize(hash_tokens, p=2, dim=-1)

            mlp_tokens = mlp_tokens.view(4 * mlp_tokens.shape[0], self.token_size // 4)
            mlp_tokens = torch.nn.functional.normalize(mlp_tokens, p=2, dim=-1)
            mlp_tokens = mlp_tokens.view(-1, self.token_size)

        

           

        rot_t = torch.zeros((self.pos_token_size, self.token_size))
        rot_m = torch.zeros_like(rot_t)
        

        #XYZ -> nn.linear position
        #(C,3) (C,128)
        hash_pos = torch.cat([points, indices_global.reshape(C,-1), indices_layers.reshape(C,-1),  indices_layerwise.reshape(C,-1)], dim=1)
        mlp_pos = torch.zeros((mlp_tokens.shape[0],hash_pos.shape[1]))
        mlp_pos[:,0] = torch.arange(0,mlp_pos.shape[0]) + self.pos_token_size
        
        rot_p = torch.zeros((self.pos_token_size,hash_pos.shape[1]))
        rot_p[:,0] = torch.arange(0,self.pos_token_size)

        return torch.vstack([hash_tokens, mlp_tokens,rot_t]), torch.vstack([hash_pos, mlp_pos,rot_p]), torch.vstack([hash_masks,mlp_masks,rot_m])
    
    def tokenize_mlp_layer(self,w):  
        mask = torch.ones_like(w)
        
        max_layer_width = self.max_layer_width
        token_size = self.token_size
        
        pad = max_layer_width - w.shape[1]
        # w - > (x , max_layer_width)
        if pad > 0:
            w = nn.functional.pad(w, (0, pad))
            mask = nn.functional.pad(mask, (0,pad))

        n_layers_per_token = (token_size // max_layer_width)
        pad_axis_0 = (n_layers_per_token - (w.shape[0] % n_layers_per_token)) % n_layers_per_token

        # (x, max_layer_width) -> (n_layers_per_token *k , max_layer_width)
        if pad_axis_0 > 0:
            w = torch.cat([w, torch.zeros(pad_axis_0, w.shape[1])], dim=0)
            mask = torch.cat([mask, torch.zeros(pad_axis_0, mask.shape[1])], dim=0)

        #(Group each 4 rows into one row)
        w = w.view(-1, n_layers_per_token , max_layer_width).view(-1,token_size)
        mask = mask.view(-1, n_layers_per_token , max_layer_width).view(-1,token_size)


        return (w,mask)
    
    import torch

    def permute_mlp_weights(self,weights):
        """
        weights: list of weight matrices, each shape (out_dim, in_dim)
        Returns:
            new_weights: list of permuted weight matrices
        """
        num_layers = len(weights)
        
        # Generate permutations for all hidden layers except input (and output)
        perms = []
        for l in range(num_layers - 1):
            out_dim = weights[l].shape[0]
            perm = torch.randperm(out_dim, device=weights[l].device)
            perms.append(perm)
        
        # Compute inverse permutations
        inv_perms = []
        for perm in perms:
            inv = torch.empty_like(perm)
            inv[perm] = torch.arange(len(perm), device=perm.device)
            inv_perms.append(inv)
        
        new_weights = []
        for l in range(num_layers):
            W = weights[l]
            if l == 0:
                # First layer: permute output rows only
                P = perms[0]
                W_new = W[P, :]
            elif l == num_layers - 1:
                # Last layer: permute input columns only (inverse of last hidden layer permutation)
                P_inv = inv_perms[-1]
                W_new = W[:, P_inv]
            else:
                # Hidden layers: permute rows and columns
                P = perms[l]
                P_inv = inv_perms[l-1]
                W_new = W[P, :]
                W_new = W_new[:, P_inv]
            new_weights.append(W_new)
        
        return new_weights

        

        

    
    # ===========================================================


from torch.utils.data import Dataset, DataLoader, random_split
import torch


class GPTText():
    def __init__(self,train, tokenizer, context_size, test = None, valid = None,  batch_size = 64, test_size = 0.1):
        super().__init__(batch_size=batch_size)
        self.context_size = context_size
        self.tokenizer = tokenizer

        self.train_dataset = self.tokenizer.encode(train)

        if test is not None:
            self.test_data = self.tokenizer.encode(test)
        else:
            if test_size is not None:
            
                self.train_size = int((1-test_size) * len(self.train_dataset))
                self.test_dataset =  self.train_dataset[self.train_size:]
                self.train_dataset = self.train_dataset[:self.train_size]

        if valid is not None:
            self.valid_dataset = self.tokenizer.encode(valid)
        
    def train_data(self):
        ix = torch.randint(len(self.train_dataset) - self.context_size, (self.batch_size,))
        x = torch.stack([torch.tensor(self.train_dataset[i: i+self.context_size]) for i in ix])
        y = torch.stack([torch.tensor(self.train_dataset[i+1 : i+1+self.context_size]) for i in ix])

        return x,y
    def test_data(self):
        ix = torch.randint(len(self.test_dataset) - self.context_size, (self.batch_size,))
        x = torch.stack([torch.tensor(self.test_dataset[i: i+self.context_size]) for i in ix])
        y = torch.stack([torch.tensor(self.test_dataset[i+1 : i+1+self.context_size]) for i in ix])

        return x,y

    def valid_data(self):
        ix = torch.randint(len(self.valid_dataset) - self.context_size, (self.batch_size,))
        x = torch.stack([torch.tensor(self.valid_dataset[i: i+self.context_size]) for i in ix])
        y = torch.stack([torch.tensor(self.valid_dataset[i+1 : i+1+self.context_size]) for i in ix])

        return x,y

    def load(self):
        print("Save for GPTText is not possible. Please save your dataset")
    def save(self):
        print("Save for GPTText is not possible. Please save your dataset")




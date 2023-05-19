import pickle as pickle
import torch


class RE_Dataset(torch.utils.data.Dataset):
    """ Dataset 구성을 위한 class."""

    def __init__(self, pair_dataset, RE_labels, S_labels, O_labels):
        self.pair_dataset = pair_dataset
        self.labels = RE_labels
        self.s_labels = S_labels
        self.o_labels = O_labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach()
                for key, val in self.pair_dataset.items()}
        if self.s_labels:
            item['labels'] = torch.tensor([self.labels[idx], self.s_labels[idx],self.s_labels[idx]])
        else:
            item['labels'] = torch.tensor(self.labels[idx])
        #item['s_labels']= torch.tensor(self.s_labels[idx])
        #item['o_labels']= torch.tensor(self.s_labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

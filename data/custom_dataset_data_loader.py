from random import shuffle
from data.base_data_loader import BaseDataLoader
import numpy as np
import jittor as jt

def CreateDataset(opt):
    dataset = None
    from data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        
        batch_size=opt.batchSize
        # sampler=train_sampler
        num_workers=int(opt.nThreads)
        self.dataset.set_attrs(batch_size=1, num_workers=num_workers)

        if opt.phase == "train":
            ### split train and validation
            self.ratio = opt.data_ratio
            # dataset_size = len(self.dataset)
            dataset_size = self.dataset.total_len-1
            # indices = list(range(dataset_size))
            # np.random.shuffle(indices)
            split = int(self.ratio * dataset_size)
            # import ipdb; ipdb.set_trace()
            # jt.dataset.SubsetRandomSampler(self.dataset, [0, split])  # self.datset is train_dataloader
            self.train_dataloader = self.dataset

            val_dataset = CreateDataset(opt).set_attrs(batch_size=batch_size, num_workers=num_workers)
            # jt.dataset.SubsetRandomSampler(val_dataset, [split, dataset_size])
            self.valid_dataloader = val_dataset


            # train_indices, val_indices = indices[:split], indices[split:]
            # train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
            # if isinstance(opt.max_dataset_size, int):
            #     import random
            #     val_indices = random.sample(val_indices, int(opt.max_dataset_size * (1-self.ratio)))
            # valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

            # self.train_dataloader = torch.utils.data.DataLoader(
            #     self.dataset,
            #     batch_size=opt.batchSize,
            #     sampler=train_sampler,
            #     num_workers=int(opt.nThreads))
            # self.valid_dataloader = torch.utils.data.DataLoader(
            #     self.dataset,
            #     batch_size=opt.batchSize,
            #     sampler=valid_sampler,
            #     num_workers=int(opt.nThreads))
        elif opt.phase == "test":
            # self.test_dataloader = torch.utils.data.DataLoader(
            #     self.dataset,
            #     batch_size=opt.batchSize,
            #     shuffle=False,
            #     num_workers=int(opt.nThreads))
            if opt.how_many < 0:
                # indices = list(range(opt.start, len(self.dataset)))
                pass
            else:
                indices = list(range(opt.start, opt.start + opt.how_many))
                DirectSequentialSampler(self.dataset, [opt.start, opt.start+opt.how_many])
            # sampler = DirectSequentialSampler(indices)
            # self.test_dataloader = torch.utils.data.DataLoader(
            #     self.dataset,
            #     batch_size=opt.batchSize,
            #     sampler=sampler,
            #     num_workers=int(opt.nThreads))
            self.test_dataloader = self.dataset

            ### split train and validation
            # self.ratio = opt.data_ratio
            # dataset_size = len(self.dataset)
            # indices = list(range(dataset_size))
            # np.random.shuffle(indices)
            # split = int(self.ratio * dataset_size)
            # train_indices, val_indices = indices[:split], indices[split:]
            # train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
            # valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

            # self.test_dataloader = torch.utils.data.DataLoader(
            #     self.dataset,
            #     batch_size=opt.batchSize,
            #     sampler=valid_sampler,
            #     num_workers=int(opt.nThreads))

    def load_data(self):
        return self.train_dataloader, self.valid_dataloader

    def load_data_test(self):
        return self.test_dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)


# class DirectSequentialSampler(torch.utils.data.Sampler):
#     r"""Samples elements sequentially, always in the same order.
#     Arguments:
#         indices: indices of dataset to sample from
#     """
#     def __init__(self, indices):
#         self.indices = indices

#     def __iter__(self):
#         return iter(self.indices)

#     def __len__(self):
#         return len(self.indices)


class DirectSequentialSampler(jt.dataset.Sampler):
    def __init__(self, dataset, indice):
        # MUST set sampler here
        dataset.sampler = self
        self.dataset = dataset
        self.indices = indice

    def __iter__(self):
        return (int(i) + self.indices[0] for i in range(self.indices[1] - self.indices[0]))

    def __len__(self):
        return self.indices[1] - self.indices[0]

# class CustomDatasetDataLoader_new(BaseDataLoader):
#     def name(self):
#         return 'CustomDatasetDataLoader_new'

#     def initialize(self, opt):
#         BaseDataLoader.initialize(self, opt)
#         self.dataset = CreateDataset(opt)
#         self.ratio = opt.data_ratio
#         dataset_size = len(self.dataset)
#         self.totalNum = dataset_size
#         indices = list(range(dataset_size))
#         split = int(self.ratio * dataset_size)
#         train_indices, val_indices = indices[:split], indices[split:]
#         train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
#         valid_sampler = DirectSequentialSampler(val_indices)

#         self.dataloader = torch.utils.data.DataLoader(
#             self.dataset,
#             batch_size=opt.batchSize,
#             sampler=valid_sampler,
#             num_workers=int(opt.nThreads))
#         self.dataset_size = len(val_indices)

#     def load_data_test(self):
#         return self.dataloader

#     def __len__(self):
#         return min(self.dataset_size , self.opt.max_dataset_size)
import jittor as jt
jt.flags.use_cuda = 1
class YourDataset(jt.dataset.Dataset):
    def __init__(self):
        super().__init__()
        self.set_attrs(total_len=32)

    def __getitem__(self, k):
        return k, k*k

from data.aligned_dataset import AlignedDataset

dataset = YourDataset().set_attrs(batch_size=8, shuffle=False, num_workers=4)
# dataset = YourDataset().set_attrs(batch_size=8, shuffle=False, num_workers=4)

# jt.dataset.SubsetRandomSampler(dataset, [0, 9])
for x, y in dataset:
#     print(x[:5], y[:5])
    print(x, y)
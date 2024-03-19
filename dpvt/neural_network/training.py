import torch
from torch.utils.data import (
    random_split,
    dataset,
    DataLoader,
)
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from dpvt.neural_network.traverse_nn import TraverseNN
from dpvt.neural_network.training_data import (
    good_trees,
    bad_trees,
    site4_good_trees,
    site4_bad_trees,
    SAMPLE_SIZE,
)


# hyperparameters
epochs = 100
TRAIN_SIZE = int(0.8 * SAMPLE_SIZE)
TEST_SIZE = SAMPLE_SIZE - TRAIN_SIZE
BATCH_SIZE = 16


class FourLeafData(dataset.Dataset):
    def __init__(self):
        self.data = good_trees + bad_trees
        self.labels = [0.0 for _ in range(24)] + [1.0 for _ in range(24)]

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


class FourLeafFourSiteData(dataset.Dataset):
    def __init__(self):
        self.data = site4_good_trees + site4_bad_trees
        self.labels = [0.0 for _ in range(SAMPLE_SIZE // 2)]
        self.labels += [1.0 for _ in range(SAMPLE_SIZE // 2)]

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


def custom_collate(items):
    """
    Args:
        items is a list of (input, output) pairs, where `input` is an ete3.Tree and
        `output` is a float
    """
    return [item[0] for item in items], torch.tensor([item[1] for item in items])


print("sample size:", SAMPLE_SIZE)
print("train size:", TRAIN_SIZE)
print("test size:", TEST_SIZE)
train_data, test_data = random_split(FourLeafFourSiteData(), [TRAIN_SIZE, TEST_SIZE])
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=custom_collate)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=custom_collate)

# use pytorch lightning
tnn = TraverseNN()
logger = TensorBoardLogger(save_dir="lightning_logs", name="TNN_4leaf_4sites_v0")
# choose how often to save model checkpoints
checkpoint_callback = ModelCheckpoint(every_n_epochs=10, save_top_k=-1)
trainer = L.Trainer(
    logger=logger,
    max_epochs=epochs,
    log_every_n_steps=1,
    callbacks=[checkpoint_callback],
)


def run():
    trainer.fit(tnn, train_loader, test_loader)


if __name__ == "__main__":
    run()

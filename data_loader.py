import os
import nltk
import torch.utils.data as data
from coco_dataset import CoCoDataset
nltk.download("punkt")

def get_loader(
    transform, 
    mode="train",
    batch_size=1,
    vocab_threshold=None,
    vocab_file="vocab.pkl",
    start_word="<start>",
    end_word="<end>",
    unk_word="<unk>",
    vocab_from_file=True,
    num_workers=0,
    cocoapi_loc="cocoapi",
):
    assert mode in ["train", "test"],"mode must be one of 'train' or 'test'"

    if not vocab_from_file:
        assert(
            mode=="train"
        ), "To generate vocab from captions file, must be in training mode (mode='train')."

    if mode == "train":
        if vocab_from_file:
            assert os.path.exists(
                vocab_file
            ), "vocab_file does not exist, Change vocab_from_file to False to create vocab_file."
        img_folder = os.path.join(cocoapi_loc, "images/train2017")
        annotations_file = os.path.join(
            cocoapi_loc, "annotations/captions_train2017.json"
        )
    elif mode == "test":
        assert batch_size == 1, "Please change batch_size to 1 if testing the model."
        assert os.path.exists(
            vocab_file
        ), "Must first generate vocab.pkl from training data."
        assert vocab_from_file, "Change vocab_from_file to True."
        img_folder = os.path.join(cocoapi_loc, "images/test2017/")
        annotations_file = os.path.join(
            cocoapi_loc, "annotations/captions_test2017.json"
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")

        #COCO captioin dataset
    dataset = CoCoDataset(
        transform=transform,
        mode=mode,
        batch_size=batch_size,
        vocab_threshold=vocab_threshold,
        vocab_file=vocab_file,
        start_word=start_word,
        end_word=end_word,
        unk_word=unk_word,
        annotations_file=annotations_file,
        vocab_from_file=vocab_from_file,
        img_folder=img_folder,
    )
    if mode == "train":
        #randomly sample a caption length and sample indices with that lenght.
        indices = dataset.get_train_indices()
        #create and assign a batch sampler to retrieve a batch with the sampled indices.
        initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        #data loader for COCO dataset.
        data_loader = data.DataLoader(
            dataset=dataset,
            num_workers=num_workers,
            batch_sampler=data.sampler.BatchSampler(
                sampler=initial_sampler, batch_size=dataset.batch_size, drop_last=False
            ),
        )
    else:
        data_loader = data.DataLoader(
            dataset=dataset,
            batch_size=dataset.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
    return data_loader
        











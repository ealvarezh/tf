import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class INat21Dataset(Dataset):
    """
    iNat21 dataset loader that returns (image, text caption, label).
    Caption: "a photo of {common_name}"
    """
    def __init__(self, root, ann_file, transform=None):
        """
        Args:
            root (str): Root directory with images (e.g. /.../inat21_image).
            ann_file (str): Path to JSON annotation file (train.json / val.json).
            transform (callable, optional): Transform to apply to images (e.g. from open_clip).
        """
        self.root = root
        self.ann_file = ann_file
        self.transform = transform

        with open(ann_file, "r") as f:
            data = json.load(f)

        self.images = data["images"]
        self.annotations = data.get("annotations", [])
        self.catid2name = {cat["id"]: cat["common_name"] for cat in data["categories"]}
        self.imgid2catid = {ann["image_id"]: ann["category_id"] for ann in self.annotations}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.root, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        img_id = img_info["id"]
        cat_id = self.imgid2catid[img_id]
        common_name = self.catid2name[cat_id]
        caption = f"a photo of {common_name}"

        return image, caption, cat_id


class INat24Dataset(Dataset):
    """
    iNat24 dataset loader — no splits, only a single folder (train/) and train.json.
    This dataset is used as a test set where:
        - images are the database
        - text (captions) are the queries
    """
    def __init__(self, root, ann_file, transform=None):
        """
        Args:
            root (str): Root directory with images (e.g. /.../inat24/train).
            ann_file (str): Path to JSON annotation file (train.json).
            transform (callable, optional): Transform to apply to images.
        """
        self.root = root
        self.ann_file = ann_file
        self.transform = transform

        with open(ann_file, "r") as f:
            data = json.load(f)

        self.images = data["images"]
        self.annotations = data.get("annotations", [])
        self.catid2name = {cat["id"]: cat["common_name"] for cat in data["categories"]}
        self.imgid2catid = {ann["image_id"]: ann["category_id"] for ann in self.annotations}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.root, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        img_id = img_info["id"]
        cat_id = self.imgid2catid[img_id]
        common_name = self.catid2name[cat_id]
        caption = f"a photo of {common_name}"

        # For iNat24, this is used as both database (image) and query (text)
        return image, caption, cat_id

def get_dataloaders(args, train_transform=None, eval_transform=None, train_drop_last=True):

    map_k = 10

    # iNat21 — train/val splits
    train_json = "train_mini.json" if getattr(args, "train_mini", False) else "train.json"
    # train_dataset = INat21Dataset(
    #     root=args.data_dir,
    #     # ann_file=os.path.join(args.data_dir, train_json),
    #     ann_file=os.path.join("..\..\data\curated", train_json),
    #     transform=train_transform,
    # )
    # eval_dataset = INat21Dataset(
    #     root=args.data_dir,
    #     ann_file=os.path.join("..\..\data\curated", "val.json"),
    #     transform=eval_transform,
    # )
    train_dataset = INat21Dataset(
        root=args.data_dir,
        ann_file=r"..\data\curated\{}".format(train_json),
        transform=train_transform,
    )

    eval_dataset = INat21Dataset(
        root=args.data_dir,
        ann_file=r"..\data\curated\val.json",
        transform=eval_transform,
)
    train_loader = DataLoader(train_dataset, batch_size=args.bs, num_workers=args.n_workers, shuffle=True, pin_memory=True, drop_last=train_drop_last)
    database_loader = DataLoader(eval_dataset, batch_size=args.bs, num_workers=args.n_workers, pin_memory=True)
    query_loader = DataLoader(eval_dataset, batch_size=args.bs, num_workers=args.n_workers, pin_memory=True)

    return train_loader, database_loader, query_loader, map_k

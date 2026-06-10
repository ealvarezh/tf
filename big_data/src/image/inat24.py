import os
import json
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader


# Map evaldataset → iNat24 supercategory name
INAT24_SUPERCATEGORIES = {
    'inat24-amphibians': 'Amphibians',
    'inat24-animalia': 'Animalia',
    'inat24-arachnids': 'Arachnids',
    'inat24-birds': 'Birds',
    'inat24-fungi': 'Fungi',
    'inat24-insects': 'Insects',
    'inat24-mammals': 'Mammals',
    'inat24-mollusks': 'Mollusks',
    'inat24-plants': 'Plants',
    'inat24-rayfish': 'Ray-finned Fishes',
    'inat24-reptiles': 'Reptiles',
}


class INat24Dataset(Dataset):
    """
    iNaturalist 2024 dataset loader with optional supercategory filtering.
    """

    def __init__(self, root, ann_file, transform=None, supercategory=None):
        self.root = root
        self.ann_file = ann_file
        self.transform = transform
        self.supercategory = supercategory

        # Load annotation file
        with open(ann_file, "r") as f:
            data = json.load(f)

        images = data["images"]
        annotations = data["annotations"]
        categories = data["categories"]

        # Map category IDs → names and supercategories
        catid2name = {c["id"]: c.get("common_name", c["name"]) for c in categories}
        catid2super = {c["id"]: c.get("supercategory", None) for c in categories}

        print(f"[INFO] Original: {len(images):,} images, {len(categories):,} categories.")

        # --- Apply supercategory filter ---
        if supercategory is not None:
            valid_cat_ids = {cid for cid, sc in catid2super.items() if sc == supercategory}
            if not valid_cat_ids:
                raise ValueError(f"No categories found for supercategory '{supercategory}'.")

            annotations = [ann for ann in annotations if ann["category_id"] in valid_cat_ids]
            valid_img_ids = {ann["image_id"] for ann in annotations}
            images = [img for img in images if img["id"] in valid_img_ids]
            categories = [c for c in categories if c["id"] in valid_cat_ids]

            print(f"[INFO] Filtered '{supercategory}': {len(images):,} images, {len(categories):,} categories.")
        else:
            print(f"[INFO] Using full dataset (no filter).")

        # Store filtered data
        self.images = images
        self.annotations = annotations
        self.categories = categories
        self.catid2name = {c["id"]: c.get("common_name", c["name"]) for c in categories}
        self.imgid2catid = {ann["image_id"]: ann["category_id"] for ann in annotations}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        file_name = img_info["file_name"]

        # Fix doubled "train/" issue
        if file_name.startswith("train/"):
            img_path = os.path.join(self.root, file_name[len("train/"):])
        else:
            img_path = os.path.join(self.root, file_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError):
            # Skip broken entries safely
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)

        img_id = img_info["id"]
        cat_id = self.imgid2catid[img_id]
        caption = f"a photo of {self.catid2name[cat_id]}"
        return image, caption, cat_id


def get_dataloaders(args, eval_transform=None):
    evaldataset = args.evaldataset.lower()
    supercategory = INAT24_SUPERCATEGORIES.get(evaldataset, None)

    dataset = INat24Dataset(
        root=os.path.join(args.data_dir, "train"),  # ✅ use correct train/ folder
        ann_file=os.path.join(args.data_dir, "train.json"),
        transform=eval_transform,
        supercategory=supercategory,
    )

    database_loader = DataLoader(dataset, batch_size=args.bs, num_workers=args.n_workers, pin_memory=True)
    query_loader = DataLoader(dataset, batch_size=args.bs, num_workers=args.n_workers, pin_memory=True)

    map_k = 1000
    return None, database_loader, query_loader, map_k
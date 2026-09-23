import os
import random
import warnings

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import timm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
)
from sklearn.utils.class_weight import compute_class_weight

try:
    from torchvision.ops import DeformConv2d
    HAS_DEFORM_CONV = True
except Exception:
    DeformConv2d = None
    HAS_DEFORM_CONV = False

warnings.filterwarnings("ignore")


# Configuration

CFG = {
    "img_size": 224,
    "batch_size": 16,
    "epochs": 100,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "num_classes": 11,
    "accumulation_steps": 1,
    "seed": 42,
    "patience": 7,
    "num_workers": 2,

    # EfficientNetV2 backbone from timm.
   
    "backbone_name": "tf_efficientnetv2_s.in21k",
    "pretrained": True,
    "out_indices": (1, 2, 3, 4),

    # Proposed module settings
    "fusion_channels": 256,
    "use_deformable": True,
    "sparse_lambda": 1e-3,
    "dropout": 0.3,
    "focal_gamma": 2.0,

    # Paths
    "best_model_path": "best_dms_efficientcapsnet_mcc.pth",
    "h5_model_path": "best_dms_efficientcapsnet_model.h5",
    "result_dir": "result",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


LABEL_PATHS = {
    "Erosion": ("erosion", "Erosion"),
    "Angiectasia": ("angiectasia", "Angiectasia"),
    "Pylorus": ("pylorus", "Pylorus"),
    "Reduced Mucosal View": ("reduced_mucosal_view", "Reduced mucosal view"),
    "Normal": ("normal_clean_mucosa", "Normal clean mucosa"),
    "Ileo-cecal valve": ("ileocecal_valve", "Ileocecal valve"),
    "Ulcer": ("ulcer", "Ulcer"),
    "Lymphangiectasia": ("lymphangiectasia", "Lymphangiectasia"),
    "Erythematous": ("erythema", "Erythema"),
    "Foreign Bodies": ("foreign_body", "Foreign body"),
    "Blood": ("blood_fresh", "Blood - fresh"),
}

ROOT = os.environ.get("KVASIR_ROOT", os.path.dirname(os.path.abspath(__file__)))

# Reproducibility

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


seed_everything(CFG["seed"])
os.makedirs(CFG["result_dir"], exist_ok=True)



# Data utilities

def build_path(row):
    folder1, folder2 = LABEL_PATHS[row.label]
    return os.path.join(ROOT, folder1, folder2, row.filename)


def add_paths_and_targets(train_csv="split_1.csv", test_csv="split_0.csv"):
    train_df = pd.read_csv(os.path.join(ROOT, train_csv))
    test_df = pd.read_csv(os.path.join(ROOT, test_csv))

    train_df["path"] = train_df.apply(build_path, axis=1)
    test_df["path"] = test_df.apply(build_path, axis=1)

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.1,
        stratify=train_df["label"],
        random_state=CFG["seed"],
    )

    # Fit on all splits so validation/test labels are always known.
    encoder = LabelEncoder()
    encoder.fit(pd.concat([train_df["label"], val_df["label"], test_df["label"]], axis=0))

    train_df["target"] = encoder.transform(train_df["label"]).astype(np.int64)
    val_df["target"] = encoder.transform(val_df["label"]).astype(np.int64)
    test_df["target"] = encoder.transform(test_df["label"]).astype(np.int64)

    return train_df, val_df, test_df, encoder


class CapsuleDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row.path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = int(row.target)
        return image, label


train_tfms = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.08), ratio=(0.3, 3.3), value=0),
])

valid_tfms = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# Model modules

class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1,
        activation=True,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.SiLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class EfficientNetV2Backbone(nn.Module):
    """
    EfficientNetV2 feature extractor using timm features_only=True.
    It returns multiple feature maps from different stages.
    """
    def __init__(self, model_name, pretrained=True, out_indices=(1, 2, 3, 4)):
        super().__init__()
        self.model_name = model_name
        self.out_indices = out_indices

        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=out_indices,
            )
        except Exception as exc:
            print(f"Could not load {model_name}: {exc}")
            print("Falling back to efficientnetv2_rw_s.")
            self.backbone = timm.create_model(
                "efficientnetv2_rw_s",
                pretrained=pretrained,
                features_only=True,
                out_indices=out_indices,
            )

        self.feature_channels = self.backbone.feature_info.channels()

    def forward(self, x):
        return self.backbone(x)


class MultiScaleCapsuleFeatureFusion(nn.Module):
    """
    Fuses EfficientNetV2 multi-level feature maps.

    Steps:
    1) Project every stage to the same channel size using 1x1 conv.
    2) Resize all projected features to a common spatial resolution.
    3) Concatenate and blend using 3x3 conv blocks.
    """
    def __init__(self, in_channels_list, out_channels=256, reference_index=1):
        super().__init__()
        self.reference_index = reference_index

        self.projections = nn.ModuleList([
            ConvBNAct(in_ch, out_channels, kernel_size=1, padding=0)
            for in_ch in in_channels_list
        ])

        self.fusion = nn.Sequential(
            ConvBNAct(out_channels * len(in_channels_list), out_channels, kernel_size=3),
            ConvBNAct(out_channels, out_channels, kernel_size=3),
        )

    def forward(self, features):
        target_size = features[self.reference_index].shape[-2:]
        aligned = []

        for feature, projection in zip(features, self.projections):
            feature = projection(feature)
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(
                    feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            aligned.append(feature)

        fused = torch.cat(aligned, dim=1)
        fused = self.fusion(fused)
        return fused


class DeformableLesionRefinementBlock(nn.Module):
    """
    Uses deformable convolution to adapt to irregular lesion shapes.
    If torchvision DeformConv2d is unavailable, it falls back to dilated convolution.
    """
    def __init__(self, channels=256, kernel_size=3, use_deformable=True, fallback_dilation=2):
        super().__init__()
        self.use_deformable = bool(use_deformable and HAS_DEFORM_CONV)
        padding = kernel_size // 2

        if self.use_deformable:
            self.offset_conv = nn.Conv2d(
                channels,
                2 * kernel_size * kernel_size,
                kernel_size=kernel_size,
                padding=padding,
            )
            self.conv = DeformConv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        else:
            self.offset_conv = None
            self.conv = nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=fallback_dilation,
                dilation=fallback_dilation,
                bias=False,
            )

        self.refine = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            ConvBNAct(channels, channels, kernel_size=1, padding=0),
        )

    def forward(self, x):
        if self.use_deformable:
            offset = self.offset_conv(x)
            y = self.conv(x, offset)
        else:
            y = self.conv(x)

        y = self.refine(y)
        return x + y


class SobelEdgeMap(nn.Module):
    """
    Generates a Sobel edge map from ImageNet-normalized RGB input.
    The image is denormalized internally before edge extraction.
    """
    def __init__(self):
        super().__init__()
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, x):
        x = (x * self.std + self.mean).clamp(0, 1)
        gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]

        edge_x = F.conv2d(gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray, self.sobel_y, padding=1)
        edge = torch.sqrt(edge_x.pow(2) + edge_y.pow(2) + 1e-6)
        edge = edge / (edge.amax(dim=(2, 3), keepdim=True) + 1e-6)
        return edge


class EdgeGuidedLesionAttention(nn.Module):
    """
    Creates an edge-based spatial attention map and uses it to enhance feature maps.
    """
    def __init__(self):
        super().__init__()
        self.edge_extractor = SobelEdgeMap()
        self.edge_attention = nn.Sequential(
            ConvBNAct(1, 32, kernel_size=3),
            ConvBNAct(32, 32, kernel_size=3),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, feature_map, input_image):
        edge_map = self.edge_extractor(input_image)
        edge_map = F.interpolate(
            edge_map,
            size=feature_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        attention = self.edge_attention(edge_map)
        enhanced = feature_map + feature_map * attention
        return enhanced, attention, edge_map


class GlobalAvgMaxPool(nn.Module):
    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, 1).flatten(1)
        max_pool = F.adaptive_max_pool2d(x, 1).flatten(1)
        return torch.cat([avg_pool, max_pool], dim=1)


class SparseFeatureSelectionGate(nn.Module):
    """
    Learnable alternative to metaheuristic feature selection.
    The gate suppresses redundant features and adds a sparsity penalty.
    """
    def __init__(self, feature_dim, reduction=4, dropout=0.2):
        super().__init__()
        hidden_dim = max(feature_dim // reduction, 16)
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        gate_values = self.gate(x)
        selected = x * gate_values
        sparsity_loss = gate_values.mean()
        return selected, sparsity_loss, gate_values


class DMSEfficientCapsNet(nn.Module):
    """
    DMS-EfficientCapsNet:
    EfficientNetV2 backbone + multi-scale fusion + deformable refinement
    + edge-guided attention + sparse feature selection.
    """
    def __init__(
        self,
        num_classes=11,
        backbone_name="tf_efficientnetv2_s.in21k",
        pretrained=True,
        out_indices=(1, 2, 3, 4),
        fusion_channels=256,
        use_deformable=True,
        dropout=0.3,
    ):
        super().__init__()

        self.backbone = EfficientNetV2Backbone(
            model_name=backbone_name,
            pretrained=pretrained,
            out_indices=out_indices,
        )

        self.multi_scale_fusion = MultiScaleCapsuleFeatureFusion(
            in_channels_list=self.backbone.feature_channels,
            out_channels=fusion_channels,
            reference_index=1,
        )

        self.deformable_refinement = DeformableLesionRefinementBlock(
            channels=fusion_channels,
            use_deformable=use_deformable,
        )

        self.edge_attention = EdgeGuidedLesionAttention()
        self.pooling = GlobalAvgMaxPool()

        pooled_dim = fusion_channels * 2
        self.feature_selector = SparseFeatureSelectionGate(
            feature_dim=pooled_dim,
            reduction=4,
            dropout=0.2,
        )

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(pooled_dim),
            nn.Dropout(dropout),
            nn.Linear(pooled_dim, num_classes),
        )

    def forward(self, x, return_aux=False):
        features = self.backbone(x)
        fused = self.multi_scale_fusion(features)
        refined = self.deformable_refinement(fused)
        edge_enhanced, edge_attention_map, edge_map = self.edge_attention(refined, x)
        pooled = self.pooling(edge_enhanced)
        selected_features, sparsity_loss, gate_values = self.feature_selector(pooled)
        logits = self.classifier(selected_features)

        if return_aux:
            return {
                "logits": logits,
                "features": selected_features,
                "sparsity_loss": sparsity_loss,
                "gate_values": gate_values,
                "edge_attention_map": edge_attention_map,
                "edge_map": edge_map,
            }

        return logits, selected_features, sparsity_loss
# Loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


# Train/evaluate functions

def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_sparse_loss = 0.0

    optimizer.zero_grad(set_to_none=True)

    for step, (images, labels) in enumerate(tqdm(loader, desc="Train", leave=False)):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True).long()

        with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            logits, _, sparse_loss = model(images)
            cls_loss = criterion(logits, labels)
            total_loss = cls_loss + CFG["sparse_lambda"] * sparse_loss
            loss = total_loss / CFG["accumulation_steps"]

        scaler.scale(loss).backward()

        should_step = ((step + 1) % CFG["accumulation_steps"] == 0) or ((step + 1) == len(loader))
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += total_loss.item()
        running_cls_loss += cls_loss.item()
        running_sparse_loss += sparse_loss.item()

    n = len(loader)
    return running_loss / n, running_cls_loss / n, running_sparse_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    preds = []
    targets = []
    running_loss = 0.0
    running_cls_loss = 0.0
    running_sparse_loss = 0.0

    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True).long()

        with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            logits, _, sparse_loss = model(images)
            cls_loss = criterion(logits, labels)
            total_loss = cls_loss + CFG["sparse_lambda"] * sparse_loss

        pred = logits.argmax(dim=1)
        preds.extend(pred.cpu().numpy())
        targets.extend(labels.cpu().numpy())

        running_loss += total_loss.item()
        running_cls_loss += cls_loss.item()
        running_sparse_loss += sparse_loss.item()

    acc = accuracy_score(targets, preds)
    balanced_acc = balanced_accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro")
    weighted_f1 = f1_score(targets, preds, average="weighted")
    mcc = matthews_corrcoef(targets, preds)

    n = len(loader)
    metrics = {
        "loss": running_loss / n,
        "cls_loss": running_cls_loss / n,
        "sparse_loss": running_sparse_loss / n,
        "acc": acc,
        "balanced_acc": balanced_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mcc": mcc,
        "targets": targets,
        "preds": preds,
    }
    return metrics


def save_model_h5(model, path):
    with h5py.File(path, "w") as f:
        for name, param in model.state_dict().items():
            safe_name = name.replace("/", "_")
            f.create_dataset(safe_name, data=param.detach().cpu().numpy())


def load_state_dict_compatible(model, path):
    try:
        state = torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)




def main():
    print(f"Using device: {DEVICE}")
    print(f"DeformConv2d available: {HAS_DEFORM_CONV}")

    train_df, val_df, test_df, encoder = add_paths_and_targets()
    CFG["num_classes"] = len(encoder.classes_)
    print(f"Detected classes ({CFG['num_classes']}): {list(encoder.classes_)}")

    missing_paths = [p for p in train_df["path"].head(10).tolist() if not os.path.exists(p)]
    if missing_paths:
        print("Warning: some example image paths do not exist. Check KVASIR_ROOT and LABEL_PATHS.")
        print(missing_paths[:3])

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df.target),
        y=train_df.target,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    train_dataset = CapsuleDataset(train_df, train_tfms)
    val_dataset = CapsuleDataset(val_df, valid_tfms)
    test_dataset = CapsuleDataset(test_df, valid_tfms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=CFG["num_workers"],
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG["batch_size"],
        shuffle=False,
        num_workers=CFG["num_workers"],
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG["batch_size"],
        shuffle=False,
        num_workers=CFG["num_workers"],
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
    )

    model = DMSEfficientCapsNet(
        num_classes=CFG["num_classes"],
        backbone_name=CFG["backbone_name"],
        pretrained=CFG["pretrained"],
        out_indices=CFG["out_indices"],
        fusion_channels=CFG["fusion_channels"],
        use_deformable=CFG["use_deformable"],
        dropout=CFG["dropout"],
    ).to(DEVICE)

    criterion = FocalLoss(alpha=class_weights, gamma=CFG["focal_gamma"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CFG["epochs"],
    )

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_mcc = -1.0
    patience_counter = 0

    history = []

    for epoch in range(CFG["epochs"]):
        print(f"\nEpoch {epoch + 1}/{CFG['epochs']}")

        train_loss, train_cls_loss, train_sparse_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
        )

        val_metrics = evaluate(model, val_loader, criterion)
        scheduler.step()

        print(f"Train Loss:        {train_loss:.4f}")
        print(f"Train Class Loss:  {train_cls_loss:.4f}")
        print(f"Train Sparse Loss: {train_sparse_loss:.4f}")
        print(f"Val Loss:          {val_metrics['loss']:.4f}")
        print(f"Val Acc:           {val_metrics['acc']:.4f}")
        print(f"Val Balanced Acc:  {val_metrics['balanced_acc']:.4f}")
        print(f"Val Macro-F1:      {val_metrics['macro_f1']:.4f}")
        print(f"Val MCC:           {val_metrics['mcc']:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_cls_loss": train_cls_loss,
            "train_sparse_loss": train_sparse_loss,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_balanced_acc": val_metrics["balanced_acc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_mcc": val_metrics["mcc"],
            "lr": optimizer.param_groups[0]["lr"],
        })
        pd.DataFrame(history).to_csv(os.path.join(CFG["result_dir"], "training_history.csv"), index=False)

        if val_metrics["mcc"] > best_mcc:
            best_mcc = val_metrics["mcc"]
            patience_counter = 0
            torch.save(model.state_dict(), CFG["best_model_path"])
            print(f"Saved new best model with MCC: {best_mcc:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= CFG["patience"]:
            print("Early stopping triggered.")
            break

    load_state_dict_compatible(model, CFG["best_model_path"])
    save_model_h5(model, CFG["h5_model_path"])
    print(f"Saved model weights to {CFG['h5_model_path']}")

    test_metrics = evaluate(model, test_loader, criterion)

    print("\nFinal Test Results")
    print("Test Accuracy:", round(test_metrics["acc"], 4))
    print("Test Balanced Accuracy:", round(test_metrics["balanced_acc"], 4))
    print("Test Macro-F1:", round(test_metrics["macro_f1"], 4))
    print("Test Weighted-F1:", round(test_metrics["weighted_f1"], 4))
    print("Test MCC:", round(test_metrics["mcc"], 4))

    report = classification_report(
        test_metrics["targets"],
        test_metrics["preds"],
        target_names=encoder.classes_,
        digits=4,
    )
    print(report)

    with open(os.path.join(CFG["result_dir"], "classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(test_metrics["targets"], test_metrics["preds"])
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=encoder.classes_,
        yticklabels=encoder.classes_,
        cmap="Blues",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("DMS-EfficientCapsNet Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(CFG["result_dir"], "confusion_matrix_dms_efficientcapsnet.png"), dpi=150)
    plt.close()

    print(f"Confusion matrix saved to {os.path.join(CFG['result_dir'], 'confusion_matrix_dms_efficientcapsnet.png')}")


if __name__ == "__main__":
    main()

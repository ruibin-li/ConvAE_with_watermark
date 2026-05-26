"""
Minimal runnable ConvAE + residual watermark training skeleton.

Main task:
    input  = x
    target = x

Watermark task for client j:
    input  = trigger(x)
    target = trigger(x) + delta_j

Run:
    python minimal_convae_residual_watermark_skeleton.py --device cuda
    python minimal_convae_residual_watermark_skeleton.py --dataset cifar10 --data-root ./data --download

This is intentionally small and hackable, not a full BlackCATT/FL implementation.
"""

import argparse
import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# -----------------------------
# 1. Small Conv Autoencoder
# -----------------------------

class ConvAE(nn.Module):
    """A small convolutional autoencoder for 32x32 images."""

    def __init__(self, in_ch: int = 3, latent_ch: int = 64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(inplace=True),
            nn.Conv2d(32, latent_ch, kernel_size=3, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_ch, latent_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_ch, 32, kernel_size=4, stride=2, padding=1),  # 8 -> 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


# -----------------------------
# 2. Watermark config and utils
# -----------------------------

@dataclass
class WMConfig:
    client_id: int = 0
    num_clients: int = 10
    trigger_size: int = 6
    trigger_value: float = 1.0
    residual_eps: float = 0.08
    wm_ratio: float = 0.25
    lambda_wm: float = 1.0
    seed: int = 1234


def make_client_residual_delta(
    client_id: int,
    num_clients: int,
    shape: tuple[int, int, int],
    eps: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Create a deterministic client-specific residual pattern delta_j.

    Shape is C x H x W. For a real paper version, this can be replaced by
    Tardos-code-like or orthogonal client codes.
    """
    c, h, w = shape
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 1009 * client_id)

    # Random sign pattern, normalized so the residual magnitude is controlled by eps.
    delta = torch.randint(0, 2, (c, h, w), generator=gen, dtype=torch.float32)
    delta = 2.0 * delta - 1.0

    # Optional: reduce high-frequency harshness by using a patch-only residual.
    mask = torch.zeros_like(delta)
    patch = min(8, h, w)
    mask[:, -patch:, -patch:] = 1.0

    delta = eps * delta * mask
    return delta.to(device)


def apply_patch_trigger(x: torch.Tensor, trigger_size: int, trigger_value: float) -> torch.Tensor:
    """
    Add a simple visible square trigger to bottom-right corner.

    x: B x C x H x W, expected in [0, 1].
    """
    x_tr = x.clone()
    x_tr[:, :, -trigger_size:, -trigger_size:] = trigger_value
    return x_tr


def make_wm_batch(
    x: torch.Tensor,
    delta_j: torch.Tensor,
    cfg: WMConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build watermark input and target.

    input:  trigger(x)
    target: trigger(x) + delta_j
    """
    x_tr = apply_patch_trigger(x, cfg.trigger_size, cfg.trigger_value)
    y_wm = torch.clamp(x_tr + delta_j.unsqueeze(0), 0.0, 1.0)
    return x_tr, y_wm


@torch.no_grad()
def residual_score(
    model: nn.Module,
    x: torch.Tensor,
    delta_j: torch.Tensor,
    cfg: WMConfig,
) -> dict[str, float]:
    """
    A simple detector-like score.

    After feeding trigger(x), check whether output - trigger(x) aligns with delta_j.
    The cosine score should increase if the model learned the watermark residual.
    """
    model.eval()
    x_tr = apply_patch_trigger(x, cfg.trigger_size, cfg.trigger_value)
    out = model(x_tr)
    residual = out - x_tr

    d = delta_j.unsqueeze(0).expand_as(residual)
    dot = (residual * d).flatten(1).sum(dim=1)
    denom = residual.flatten(1).norm(dim=1) * d.flatten(1).norm(dim=1) + 1e-8
    cos = dot / denom

    mse_to_wm_target = F.mse_loss(out, torch.clamp(x_tr + delta_j.unsqueeze(0), 0.0, 1.0)).item()
    residual_l2 = residual.flatten(1).norm(dim=1).mean().item()

    return {
        "wm_cos": cos.mean().item(),
        "wm_target_mse": mse_to_wm_target,
        "residual_l2": residual_l2,
    }


# -----------------------------
# 3. Data
# -----------------------------

def build_dataset(name: str, root: str, download: bool, image_size: int):
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    if name == "fake":
        return datasets.FakeData(
            size=5000,
            image_size=(3, image_size, image_size),
            num_classes=10,
            transform=transforms.ToTensor(),
        ), 3

    if name == "cifar10":
        return datasets.CIFAR10(root=root, train=True, download=download, transform=tfm), 3

    if name == "mnist":
        mnist_tfm = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        return datasets.MNIST(root=root, train=True, download=download, transform=mnist_tfm), 1

    raise ValueError(f"Unknown dataset: {name}")


# -----------------------------
# 4. Train / eval loop
# -----------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    delta_j: torch.Tensor,
    cfg: WMConfig,
    device: torch.device,
) -> dict[str, float]:
    model.train()

    total_main = 0.0
    total_wm = 0.0
    total = 0.0
    n_batches = 0

    for x, _ in loader:
        x = x.to(device)

        # Main reconstruction loss.
        out_main = model(x)
        loss_main = F.mse_loss(out_main, x)

        # Use a subset of each batch for watermark training.
        wm_bs = max(1, int(math.ceil(cfg.wm_ratio * x.size(0))))
        x_wm_base = x[:wm_bs]
        x_wm, y_wm = make_wm_batch(x_wm_base, delta_j, cfg)
        out_wm = model(x_wm)
        loss_wm = F.mse_loss(out_wm, y_wm)

        loss = loss_main + cfg.lambda_wm * loss_wm

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_main += loss_main.item()
        total_wm += loss_wm.item()
        total += loss.item()
        n_batches += 1

    return {
        "loss": total / n_batches,
        "main_mse": total_main / n_batches,
        "wm_mse": total_wm / n_batches,
    }


@torch.no_grad()
def evaluate_reconstruction(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 10,
) -> float:
    model.eval()
    losses = []
    for b, (x, _) in enumerate(loader):
        if b >= max_batches:
            break
        x = x.to(device)
        out = model(x)
        losses.append(F.mse_loss(out, x).item())
    return sum(losses) / max(1, len(losses))


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, default="fake", choices=["fake", "cifar10", "mnist"])
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--download", action="store_true")
    p.add_argument("--image-size", type=int, default=32)

    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-ch", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--client-id", type=int, default=0)
    p.add_argument("--num-clients", type=int, default=10)
    p.add_argument("--trigger-size", type=int, default=6)
    p.add_argument("--residual-eps", type=float, default=0.08)
    p.add_argument("--wm-ratio", type=float, default=0.25)
    p.add_argument("--lambda-wm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--save-path", type=str, default="convae_wm.pt")

    return p.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    dataset, in_ch = build_dataset(args.dataset, args.data_root, args.download, args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = ConvAE(in_ch=in_ch, latent_ch=args.latent_ch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cfg = WMConfig(
        client_id=args.client_id,
        num_clients=args.num_clients,
        trigger_size=args.trigger_size,
        residual_eps=args.residual_eps,
        wm_ratio=args.wm_ratio,
        lambda_wm=args.lambda_wm,
        seed=args.seed,
    )

    delta_j = make_client_residual_delta(
        client_id=cfg.client_id,
        num_clients=cfg.num_clients,
        shape=(in_ch, args.image_size, args.image_size),
        eps=cfg.residual_eps,
        seed=cfg.seed,
        device=device,
    )

    print("Config:")
    print(f"  dataset      = {args.dataset}")
    print(f"  device       = {device}")
    print(f"  client_id    = {cfg.client_id}")
    print(f"  wm_ratio     = {cfg.wm_ratio}")
    print(f"  lambda_wm    = {cfg.lambda_wm}")
    print(f"  residual_eps = {cfg.residual_eps}")
    print()

    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(model, loader, optimizer, delta_j, cfg, device)
        recon_mse = evaluate_reconstruction(model, eval_loader, device)

        x_probe, _ = next(iter(eval_loader))
        x_probe = x_probe.to(device)
        wm_stats = residual_score(model, x_probe, delta_j, cfg)

        print(
            f"Epoch {epoch:03d} | "
            f"loss={stats['loss']:.6f} | "
            f"main_mse={stats['main_mse']:.6f} | "
            f"wm_mse={stats['wm_mse']:.6f} | "
            f"eval_recon_mse={recon_mse:.6f} | "
            f"wm_cos={wm_stats['wm_cos']:.4f} | "
            f"wm_target_mse={wm_stats['wm_target_mse']:.6f} | "
            f"residual_l2={wm_stats['residual_l2']:.4f}"
        )

    ckpt = {
        "model": model.state_dict(),
        "args": vars(args),
        "wm_config": cfg.__dict__,
        "delta_j": delta_j.detach().cpu(),
    }
    torch.save(ckpt, args.save_path)
    print(f"Saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()

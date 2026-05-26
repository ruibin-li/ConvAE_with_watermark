import argparse
import torch
from torch.utils.data import DataLoader

from minimal_convae_residual_watermark_skeleton import (
    ConvAE,
    WMConfig,
    build_dataset,
    make_client_residual_delta,
    residual_score,
)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--dataset", type=str, default="cifar10")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num-eval-clients", type=int, default=10)
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    train_args = ckpt["args"]

    image_size = train_args.get("image_size", 32)
    latent_ch = train_args.get("latent_ch", 64)
    trained_client_id = train_args.get("client_id", 0)
    num_clients = train_args.get("num_clients", args.num_eval_clients)

    dataset, in_ch = build_dataset(
        args.dataset,
        args.data_root,
        download=False,
        image_size=image_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    x_probe, _ = next(iter(loader))
    x_probe = x_probe.to(device)

    model = ConvAE(in_ch=in_ch, latent_ch=latent_ch).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    cfg = WMConfig(
        client_id=trained_client_id,
        num_clients=num_clients,
        trigger_size=train_args.get("trigger_size", 6),
        trigger_value=train_args.get("trigger_value", 0.5),
        residual_eps=train_args.get("residual_eps", 0.12),
        wm_ratio=train_args.get("wm_ratio", 0.25),
        lambda_wm=train_args.get("lambda_wm", 3.0),
        seed=train_args.get("seed", 1234),
    )

    print(f"Trained client id: {trained_client_id}")
    print()
    print("candidate_client | wm_cos | wm_target_mse | residual_l2")
    print("-" * 58)

    for cid in range(args.num_eval_clients):
        delta = make_client_residual_delta(
            client_id=cid,
            num_clients=num_clients,
            shape=(in_ch, image_size, image_size),
            eps=cfg.residual_eps,
            seed=cfg.seed,
            device=device,
        )

        stats = residual_score(model, x_probe, delta, cfg)
        mark = "<-- correct" if cid == trained_client_id else ""

        print(
            f"{cid:16d} | "
            f"{stats['wm_cos']:6.4f} | "
            f"{stats['wm_target_mse']:13.6f} | "
            f"{stats['residual_l2']:11.4f} {mark}"
        )

if __name__ == "__main__":
    main()

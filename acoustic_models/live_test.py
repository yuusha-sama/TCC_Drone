from __future__ import annotations
import argparse, time
import sounddevice as sd
from .adapter import AcousticNNModel

def list_devices():
    for i, d in enumerate(sd.query_devices()):
        print(f"{i:2d} | {d['name']}")

def main():
    ap = argparse.ArgumentParser(description="Teste ao vivo (Keras)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--labels", default="models/labels.txt")
    ap.add_argument("--rate", type=int, default=16000)
    ap.add_argument("--win", type=float, default=1.0)
    ap.add_argument("--hop", type=float, default=0.5)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--idx_drone", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        list_devices(); return

    nn = AcousticNNModel(
        model_path=args.model, labels_path=args.labels,
        n_fft=args.n_fft, n_mels=args.n_mels,
        sample_rate=args.rate, win_s=args.win, hop_s=args.hop,
        idx_drone=args.idx_drone, device_index=args.device, threshold=args.threshold
    )
    def on_res(r: dict):
        p = float(r.get("p_drone", 0.0))
        print(f"p(drone)={p:.3f}  => {'DRONE' if p>=args.threshold else '---'}")

    print(">> Escutando… (Ctrl+C para sair)")
    nn.start(on_res)
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        nn.stop()

if __name__ == "__main__":
    main()

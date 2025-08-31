# PYNQ inference script for plant disease detection with hls4ml accelerator on Zybo Z7-10
# Usage on board:
#   python3 pynq_infer.py --bit system.bit --hwh system.hwh --labels classes.txt --image test.jpg

import argparse
import numpy as np
from PIL import Image
from pynq import Overlay, allocate

IMG_SIZE = 224


def preprocess(image_path: str):
    img = Image.open(image_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img).astype(np.float32) / 255.0
    # NHWC -> NCHW or flat depending on hls4ml interface; we assume flat float input
    return arr.flatten()


def softmax(x):
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bit', required=True)
    ap.add_argument('--hwh', required=True)
    ap.add_argument('--image', required=True)
    ap.add_argument('--labels', default='classes.txt')
    args = ap.parse_args()

    # Load overlay
    ov = Overlay(args.bit)
    ov.read(ov.bitstream_path)

    # For simplicity, we assume AXI DMA named 'axi_dma_0' is present
    dma = ov.axi_dma_0

    x = preprocess(args.image)
    in_buf = allocate(shape=(x.size,), dtype=np.float32)
    out_buf = allocate(shape=(10,), dtype=np.float32)  # adjust to number of classes
    in_buf[:] = x

    # Transfer and run
    dma.sendchannel.transfer(in_buf)
    dma.recvchannel.transfer(out_buf)
    dma.sendchannel.wait()
    dma.recvchannel.wait()

    probs = softmax(out_buf)
    labels = []
    try:
        with open(args.labels) as f:
            labels = [l.strip() for l in f]
    except Exception:
        labels = [f'class_{i}' for i in range(len(probs))]

    top = int(np.argmax(probs))
    print('Prediction:', labels[top], 'prob', float(probs[top]))
    for i, p in enumerate(probs):
        print(i, labels[i] if i < len(labels) else f'class_{i}', float(p))


if __name__ == '__main__':
    main()

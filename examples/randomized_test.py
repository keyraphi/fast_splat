from fast_splat_2d import splat
import torch
import matplotlib.pyplot as plt
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("N_PATCHES", type=int, default=1000)
    args = parser.parse_args()

    device = "cpu"
    target = torch.zeros([1080, 1440, 3], device=device)
    # create random patches with random center positions
    patches = torch.rand([args.N_PATCHES, 10, 10, 3], device=device)
    positions = torch.rand([args.N_PATCHES, 2], device=device)
    positions[:, 0] *= 1440
    positions[:, 1] *= 1080
    # run splatting on CPU
    result_cpu = splat(patches, positions, target)
    result_cpu = torch.from_dlpack(result_cpu)

    # run splatting on GPU by moving input data to GPU
    device = "cuda:0"
    target = target.to(device)
    patches = patches.to(device)
    positions = positions.to(device)
    result_gpu = splat(patches, positions, target)
    result_gpu = torch.from_dlpack(result_gpu)

    print(f"CPU result sum: {result_cpu.sum()}, GPU result sum: {result_gpu.sum()}")

    fig_cpu = plt.figure()
    ax = fig_cpu.add_subplot(111)
    ax.imshow(result_cpu)
    ax.set_title("cpu")
    fig_gpu = plt.figure()
    ax = fig_gpu.add_subplot(111)
    ax.imshow(result_gpu.cpu())
    ax.set_title("gpu")
    fig_diff = plt.figure()
    ax = fig_diff.add_subplot(111)
    ax.imshow(torch.abs(result_cpu - result_gpu.cpu()))
    ax.set_title("diff")

    print("Difference sum:", torch.abs(result_cpu - result_gpu.cpu()).sum())
    print("Max absolute difference:", torch.max(torch.abs(result_cpu - result_gpu.cpu())))


if __name__ == "__main__":
    main()

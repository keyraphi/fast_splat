from fast_splat_2d import splat
import torch
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("N_PATCHES", type=int, default=1000)
    parser.add_argument("--targe_width", type=int, default=1440)
    parser.add_argument("--targe_height", type=int, default=1080)
    args = parser.parse_args()

    device = "cpu"
    target = torch.zeros([3, 1080, 1440], device=device)
    # create random patches with random center positions
    patches = torch.rand([args.N_PATCHES, 3, 10, 10], device=device)
    positions = torch.rand([2, args.N_PATCHES], device=device)
    positions[0, :] *= 1440
    positions[1, :] *= 1080
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

    print("Difference sum:", torch.abs(result_cpu - result_gpu.cpu()).sum())
    print("Max absolute difference:", torch.max(torch.abs(result_cpu - result_gpu.cpu())))


if __name__ == "__main__":
    main()

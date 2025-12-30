from time import time
import imageio.v3 as iio
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
import fast_splat_2d
import torch
from tqdm.auto import tqdm


def create_circles_of_confusion(circle_radius_list: np.ndarray, max_blur_px: int):
    radius = np.maximum(circle_radius_list - 0.5, 1e-6)
    patches = np.zeros([radius.shape[0], 2 * max_blur_px + 1, 2 * max_blur_px + 1])

    # create patches that contain the pixel distances to the center of the patch
    line = np.linspace(
        np.floor(-max_blur_px), np.ceil(max_blur_px), 2 * int(np.ceil(max_blur_px)) + 1
    )
    xs, ys = np.meshgrid(line, line)
    points = np.stack([ys, xs], axis=0)
    distances_patch = np.linalg.norm(points, axis=0)
    patches[:] = distances_patch

    # use actual radii to mask away distances that are further away
    patches = patches < radius[:, None, None]
    patches = patches.astype(np.float32)
    patches = patches / np.sum(np.sum(patches, axis=-1), axis=-1)[:, None, None]
    return patches


def create_circles_of_confusion_gpu(circle_radius_list: torch.Tensor, max_blur_px: int):
    device = circle_radius_list.device
    radii = circle_radius_list

    radius_clamped = torch.maximum(radii - 0.5, torch.tensor(1e-6, device=device))

    steps = 2 * max_blur_px + 1
    line = torch.linspace(-max_blur_px, max_blur_px, steps, device=device)

    ys, xs = torch.meshgrid(line, line, indexing="ij")

    points = torch.stack([ys, xs], dim=0)
    dist_patch = torch.linalg.vector_norm(points, dim=0)

    patches = dist_patch[None, :, :] < radius_clamped[:, None, None]
    patches = patches.to(torch.float32)


    sums = torch.sum(patches, dim=(1, 2), keepdim=True)

    patches = patches / torch.clamp(sums, min=1e-8)

    return patches


# some tonemappiung
def reinhard(img):
    img = img / (1 + img)
    # Apply gamma
    return np.clip(img, 0, 1) ** (1 / 2.2)


# some more tonemapping
def aces_approx(x):
    # Standard fitted constants for ACES filmic curve
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    # The curve
    res = (x * (a * x + b)) / (x * (c * x + d) + e)
    return np.clip(res, 0, 1) ** (1 / 2.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--focus_distance",
        type=float,
        default=11,
        help="Focus distance of camera in meter",
    )
    parser.add_argument(
        "--f_number",
        type=float,
        default=1.4,
        help="F-number on the lens. Determines aperture radius.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Number of patches to splat simultaneously. This is limiting the memory that is used.",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="If this flag is set the splatting is done on GPU.",
    )

    args = parser.parse_args()

    img = iio.imread("scene.exr", plugin="opencv", flags=cv2.IMREAD_UNCHANGED)[:, :, :3]
    img = np.ascontiguousarray(img.transpose([2, 0, 1]))
    # img = img * 0.5
    depth = iio.imread("depth.exr", plugin="opencv", flags=cv2.IMREAD_UNCHANGED)[
        :, :, 0
    ]

    focal_length = 25.5 * 1e-3  # 25.5 mm lens was used
    f_number = args.f_number
    aperture_radius = focal_length / (2 * f_number)
    horizontal_film_size = 0.036  # 36 mm
    pixel_size = horizontal_film_size / img.shape[2]
    focus_distance = args.focus_distance

    # lens maker equations
    distance_senor_lens = 1 / (1 / focal_length - 1 / focus_distance)

    blur_radius = (
        np.abs(distance_senor_lens - 1 / (1 / focal_length - 1 / depth))
        * (1 / focal_length - 1 / depth)
        * aperture_radius
    )
    blur_radius_px = blur_radius / pixel_size
    max_blur_px = int(np.ceil(np.max(blur_radius_px)))

    print(
        f"max splat radius: {max_blur_px} => patch sizes: ({max_blur_px * 2 + 1}, {max_blur_px * 2 + 1})"
    )
    fig_blur_radius = plt.figure(figsize=(6, 6))
    blur_radius_ax = fig_blur_radius.add_subplot(111)
    blur_radius_ax.imshow(blur_radius_px)
    blur_radius_ax.set_title("circle of confusion radius")
    fig_blur_radius.show()

    # Compute patches in batches of fixed size and add them to result incrementally
    batch_size = args.batch_size
    n_batches = int(np.ceil(img.shape[1] * img.shape[2] / batch_size))
    indices = np.arange(img.shape[1] * img.shape[2])
    # np.random.shuffle(indices)
    batch_indices = np.array_split(indices, n_batches)

    duration_circle_creation = 0
    duration_splatting = 0
    pixel_list = np.reshape(img, [3, -1]).transpose()
    blur_radius_px_list = np.reshape(blur_radius_px, [-1])

    xs, ys = np.meshgrid(np.arange(img.shape[2]), np.arange(img.shape[1]))
    pixel_coordinates = np.stack([xs, ys], axis=0)
    pixel_coordinates = pixel_coordinates.reshape([2, -1])

    if args.use_gpu:
        # device = "cpu"
        device = "cuda:0"
        blur_radius_px_list = torch.as_tensor(blur_radius_px_list, device=device)
        pixel_coordinates = torch.as_tensor(pixel_coordinates, device=device)
        pixel_list = torch.as_tensor(pixel_list, device=device)
        result_image = torch.zeros(
            [img.shape[0], img.shape[1], img.shape[2]],
            dtype=torch.float32,
            device=device,
        )
        for indices in tqdm(batch_indices):
            indices = torch.as_tensor(indices, device=device)
            # create patches to splat
            time_before_circle_creation = time()
            patch_list_batch = create_circles_of_confusion_gpu(
                blur_radius_px_list[indices], max_blur_px
            )
            patch_list_batch = (
                patch_list_batch[:, None, :, :] * pixel_list[indices, :, None, None]
            )
            position_list_batch = pixel_coordinates[:, indices]
            # torch.cuda.synchronize()
            duration_circle_creation += time() - time_before_circle_creation

            # splat
            time_before_splatting = time()
            result_image = fast_splat_2d.splat(
                patch_list_batch, position_list_batch, result_image
            )
            # torch.cuda.synchronize()
            duration_splatting += time() - time_before_splatting
        result_image = torch.from_dlpack(result_image).cpu().numpy()
    else:
        result_image = np.zeros_like(img)
        for indices in tqdm(batch_indices):
            # create patches to splat
            time_before_circle_creation = time()
            patch_list_batch = create_circles_of_confusion(
                blur_radius_px_list[indices], max_blur_px
            )
            patch_list_batch = (
                patch_list_batch[:, None,:, : ] * pixel_list[indices,:, None, None]
            )
            position_list_batch = pixel_coordinates[:, indices]
            duration_circle_creation += time() - time_before_circle_creation

            # splat
            time_before_splatting = time()
            result_image = fast_splat_2d.splat(
                patch_list_batch, position_list_batch, result_image
            )
            duration_splatting += time() - time_before_splatting
        result_image = np.from_dlpack(result_image)

    print(f"Creating the patches took {duration_circle_creation} sec.")
    print(f"Splatting took {duration_splatting} sec.")

    img = img.transpose([1,2,0])
    result_image = result_image.transpose([1,2,0])
    # Show result
    fig_img = plt.figure(figsize=(12, 6))
    ax_sharp = fig_img.add_subplot(121)
    ax_sharp.imshow(reinhard(img * 0.5))
    ax_sharp.set_title("Original")

    ax_result = fig_img.add_subplot(122, sharex=ax_sharp, sharey=ax_sharp)
    ax_result.imshow(reinhard(result_image * 0.5))
    ax_result.set_title("With DoF")
    fig_img.tight_layout()
    fig_img.show()

    plt.show()

    print("Writing out images")
    iio.imwrite(
        f"results/circles_of_confusion_fd_{args.focus_distance}_f_{args.f_number}.exr",
        blur_radius_px,
    )
    input_image_8 = (reinhard(img * 0.5) * 255).astype(np.uint8)
    iio.imwrite(f"results/input.png", input_image_8)
    result_image_8 = (reinhard(result_image * 0.5) * 255).astype(np.uint8)
    iio.imwrite(
        f"results/result_fd_{focus_distance}_f_{args.f_number}.png", result_image_8
    )


if __name__ == "__main__":
    main()

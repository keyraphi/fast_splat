# Fast Splat 2D

A simple and fast function for splatting a batch of small images into a target image. It is implemented in C++ and CUDA and bound to Python for easy use from PyTorch, numpy or similar.

## Installation
Download the correct wheel for your system from the release page.
Install to your currently active environment via
```bash
pip install <the downloaded wheel>
```

Confirm that the installation worked:
```
python -c "from fast_splat_2d import splat; help(splat)"
```


## Usage
The `fast_splat_2d` module only contains one function `splat(patch_list, position_list, target_image)`.
`patch_list` are the patches that are splatted to the `target_image`. `position_list` holds the center position of the patches in the target image.

If all inputs are on host memory `splat` is executed on CPU. If all inputs are on a CUDA device
`splat` is executed on GPU. The GPU implementation is much faster.

The result is returned as a dlpack array, which can be converted to tensors in any major tensor processing library such as numpy, torch or jax.

```python
# See examples/simple_usage.py
from fast_splat_2d import splat
import torch

def main():
    ## Creating the inputs ###############################
    # Creating the splat target image [height, width, rgb]
    target = torch.zeros([123, 321, 3])

    # Create patches to splat into the target image
    # The patches all have to have the same size [n_patches, patch_height, patch_width, rgb]
    # (patch_height can be different to patch_width)
    patches = torch.rand([100, 42, 42, 3])

    # Specify position of each patch (float)
    # positions[:, 0] are the X-coordinates
    # positions[:, 1] are the Y-coordinates
    positions = torch.rand([100, 2])
    positions[:, 0] *= 321  # x-coordinate between 0 and 321
    positions[:, 1] *= 123  # y-coordinate betewen 0 and 123


    ## Splat the patches into the target image ##############
    result = splat(patches, positions, target)


    ## Repeated splatting
    # You can splat more patches into the same image by using the result as new target.
    # This can be used to batch the splatting process.
    new_patches = torch.rand([500, 42, 42, 3])
    new_patchespositions = torch.rand([100, 2])
    new_patchespositions[:, 0] *= 321
    new_patchespositions[:, 1] *= 123
    # Splat using the previous result as target
    result = splat(patches, positions, result)



    ## The result is a dlpack tensor which can be converted to numpy, torch, jax, ...
    torch_result = torch.from_dlpack(result)


    ## Do the same on GPU by having all input tensors on GPU already (much faster)
    if not torch.cuda.is_available():
        print("No cuda available")
        return
    target = torch.zeros([123, 321, 3], device="cuda:0")

    # Create patches to splat into the target image
    # The patches all have to have the same size [n_patches, patch_height, patch_width, rgb]
    # (patch_height can be different to patch_width)
    patches = torch.rand([100, 42, 42, 3], device="cuda:0")

    # Specify position of each patch (float)
    # positions[:, 0] are the X-coordinates
    # positions[:, 1] are the Y-coordinates
    positions = torch.rand([100, 2], device="cuda:0")
    positions[:, 0] *= 321  # x-coordinate between 0 and 321
    positions[:, 1] *= 123  # y-coordinate betewen 0 and 123

    # Splat the patches into the target image
    result = splat(patches, positions, target)

    # NOTE: The GPU implementation performs best when the patches are uniformly distributed
    #       over the target image.


if __name__ == "__main__":
    main()
```

## Examples
See the `examples/` directory for some usage examples.

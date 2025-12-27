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



    ## The result is a dlpack tensor wich can be converted to numpy, torch, jax, ...
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

# Examples
Some usage examples for `fast_splat_2d`.

## simple_usage.py
This demonstrates how to use the splatting function correctly. This is the code that is shown on the main README.

## randomized_test.py
This is very similar to the `simple_usage.py`, except that the results from the CPU and GPU computations are compared.
The maximum absolute difference between pixels in the cpu result and gpu result are in the order of `1e-4`.

## dept_of_field.py
This is a realistic usecase for the splatting function, simulating the depth of field effect of a lens.
As input it uses a scene rendered with a pinhole camera and a corresponding depth map.
![input image](results/input.png "Input Image")
![depth map](results/depth_map.png "Depth Map")

Based on the given lens characteristics (focus distance and f-number) it
computes for each pixels depth how much out of focus it would be. 
A scene point at a depth that is out of focus is projected onto the sensor not as a singe
dot, but as a circle. The radius of this circle is computed for every single pixel in the depth map.  
![circle of confusion](results/f_0.2/circles_of_confusion_fd_3.47_f_0.2.png "Splat Radii")

This is then used to create one patch for every pixel in the input image that contains a circle with the corresponding radius and the color of the pixel.
Those patches are than splatted onto a black target image to create the result image with the depth of field effect:
![result](results/f_0.2/result_fd_3.47_f_0.2.png "Result with depth of field")

### Usage

```bash
usage: depth_of_field.py [-h] [--focus_distance FOCUS_DISTANCE] [--f_number F_NUMBER] [--batch_size BATCH_SIZE] [--use_gpu]

options:
  -h, --help            show this help message and exit
  --focus_distance FOCUS_DISTANCE
                        Focus distance of camera in meter
  --f_number F_NUMBER   F-number on the lens. Determines aperture radius.
  --batch_size BATCH_SIZE
                        Number of patches to splat simultaneously. This is limiting the memory that is used.
  --use_gpu             If this flag is set the splatting is done on GPU.
```



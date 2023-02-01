
# Image Representation and Reconstruction by Compositing Gaussian Ellipses
*Chang-Chieh Cheng, ITSC, NYCU, Taiwan*

This study proposes a method of stroke-based rendering for image representation and reconstruction.
The proposed method involves compositing a set of ellipses by combining various locations, sizes, rotation angles, colors, and opacities.
This study also prove that alpha compositing is differentiable if the two-dimensional Gaussian function is used to draw a solid ellipse.
The gradient method can then be employed to automatically identify the parameters of each ellipse such that the difference between the input image and the composited image is minimized.
The experimental results indicate that the proposed method can represent various types of images of diverse subjects.
The proposed method can also be applied to painting style simulation and sparse-view computed tomography imaging.

vgi/imaging.py includes all implementation code of the proposed method, where decomposite() is the primary function.



# Image Representation and Reconstruction by Compositing Gaussian Ellipses
**Accepted by *IET Image Processing*, Oct. 2023.**

*Chang-Chieh Cheng, ITSC, NYCU, Taiwan*

This study proposes a method of stroke-based rendering is proposed for image representation and reconstruction. The proposed method involves compositing a set of ellipses that greatly vary in location, size, rotation angle, color, and opacity. This study proves that alpha compositing is differentiable if a two-dimensional Gaussian function is used to draw a solid ellipse. The gradient method can then be employed for automatically identifying the parameters of each ellipse such that the difference between the input image and the composited image is minimized. Experimental results indicate that the proposed method can represent various types of images including portraits, landscapes, buildings, street scenes, artificial objects, medical images, etc. The proposed method can particularly render the most details and significant features of an input image with fewer strokes compared to other stroke-based rendering algorithms. This study also demonstrates that the proposed method can be applied in painting style simulation and sparse-view computed tomography imaging.

vgi/imaging.py includes all implementation code of the proposed method, where decomposite() is the primary function.

The proposed method can also be applied to sparse-view computed tomography imaging. See [github.com/jameschengcs/gect](https://github.com/jameschengcs/gect)


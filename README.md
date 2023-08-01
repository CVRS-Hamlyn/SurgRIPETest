# SurgRIPETest

### sample.py
sample is the example of how to use the evaluate script. To use the evaluation script, model and image_reader() need be customized.

The model input is the RGB image and the ideal output should be a [4*3] matrix consist of the rotation matrix(R)[3*3] and translation matrix(T)[3]
The pose format should be the same as the GT pose as shown in visualization.py.

### evaluate.py
evaluate.py includes all the evaluateion metrics, including ADD, Translation/Rotation Error, 2d projections, 5 mm 5-degree metric, etc.

### visualization.py
visualization.py includes the visualisation methods. It provides some sample code about how to read image and annotation, how to decompose the 6DoF pose, how to show axis of pose, etc.
# SurgRIPETest

This is the official repository for the SurgRIPE competition. The challenge is hosted [on Synapse](https://www.synapse.org/#!Synapse:syn51471789/wiki/622255). Please click the link for more details.

The challange is 6 degrees of freedom tool pose estimation with and without occulusion.

## Docker Template

We have provided an example of a docker image to run this code.

(example, below should change to the docker template)
```yaml
FROM anibali/pytorch:2.0.0-cuda11.8-ubuntu22.04

# Set up time zone.
ENV TZ=UTC
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

# Install system libraries required by OpenCV.
RUN sudo apt-get update \
 && sudo apt-get install -y libgl1-mesa-glx libgtk2.0-0 libsm6 libxext6 \
 && sudo rm -rf /var/lib/apt/lists/*

# Install requirements.
RUN pip install -r requirements.txt

# run file
RUN python sample.py --path [path] --task [task]

```

### Example

```bash
docker run ...
```

For docker help, please see there [documentation](https://docs.docker.com) for more.

## sample.py
sample is the tempate of how to run the evaluation script. To use the evaluation script, model and image_reader() need be customized.

```python
def image_reader(img_path):
    ### TODO
    img = None
    return img
...
model = None
```

The model input is the RGB image and the ideal output should be a [4*3] matrix consist of the rotation matrix(R)[3*3] and translation matrix(T)[3]
The pose format should be the same as the GT pose as shown in visualization.py.

### Usage

```bash
usage: python sample.py [--path] [--type]

positional arguments:
Path                The path to root of the dataset
Type                The type of task l (LND) | m (MBF)
```

## evaluate.py
evaluate.py includes all the evaluateion metrics, including ADD, Translation/Rotation Error, 2d projections, 5 mm 5-degree metric, etc.

## visualization.py
visualization.py includes the visualisation methods. It provides some sample code about how to read image and annotation, how to decompose the 6DoF pose, how to show axis of pose, etc.
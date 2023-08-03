# SurgRIPETest

![pose](assets/pose_1.png)

This is the official repository for the MICCAI 2023 SurgRIPE competition. The challenge is hosted [on Synapse](https://www.synapse.org/#!Synapse:syn51471789/wiki/622255).

The challange - 6 DOF pose estimation for surgical tools, with and without occulusion.

A docker image is required for submission to this challenge. Participants are expected to: (1) create a docker image; (2) download the SurgeRIPETest GitHub repository; (3) modify the model initislisation and the dataloading in sample.py.


## Docker Template

Below is a template docker file that can be used. The docker file can be customised for your setup - PROVIDED it is setup to run the evaluation script! 

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

For docker help - [documentation](https://docs.docker.com).

## sample.py
sample.py contains the template of how to run the evaluation script. Before running, please modify the following in evaluate.py: model [line #34] and image_reader() [line #8].

```python
def image_reader(img_path):
    ### TODO
    img = None
    return img
...
model = None
```

The model input is an RGB image and the required output should be a [4 X 3] matrix consisting of a rotation matrix(R)[3 X 3] and a translation matrix(T)[3]
The pose format should be the same as the GT pose, as shown in visualization.py.

### Usage

```bash
usage: python sample.py [--path] [--type]

positional arguments:
Path                The path to root of the dataset
Type                The type of task l (LND) | m (MBF)
```

## evaluate.py
evaluate.py includes all the evaluateion metrics, including: ADD, Translation/Rotation Error, 2d projections, 5 mm 5-degree metric, etc.

## visualization.py
visualization.py includes the visualisation methods. Providing sample code for: how to read an image and annotation, how to decompose the 6DoF pose, how to show axis of pose, etc.

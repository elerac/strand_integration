# Multiview Data

## Directory Structure

The multiview data is stored in a folder, and each subfolder contains the data of a single view. The folder
must be named as the number of the view, like `00`, `01`, `02`, ...

```shell
.
├── mydata
│   ├── 00
│   ├── 01
│   ├── 02
│   ├── 03
```

The single view data contains images and camera parameters. The minimum requirement is the intensity image (`intensity.exr`) and the camera patameters (`K.txt`, `R.txt`, `t.txt`).

```shell
.
├── mydata
│   ├── 00
│   │   ├── intensity.exr
│   │   ├── K.txt
│   │   ├── R.txt
│   │   ├── t.txt
│   ├── 01
│   │   ├── intensity.exr
│   │   ├── K.txt
│   │   ├── R.txt
│   │   ├── t.txt
```

Some of extra images such as 2D orientation image (`orientation2d.exr`) and confidence image (`confidence.exr`) can be also stored in the subfolder. Additionally, you can also store the depth image (`depth.exr`) and direction image (`direction.exr`) in the subfolder to restore 3D lines.

```shell
.
├── mydata
│   ├── 00
│   │   ├── intensity.exr
│   │   ├── K.txt
│   │   ├── R.txt
│   │   ├── t.txt
│   │   ├── orientation2d.exr
│   │   ├── confidence.exr
│   │   ├── depth.exr
│   │   ├── direction.exr
│   ├── 01
```

The name of the filenames are defined in `utils/configs.py`. You can change the filenames by editing this file.

## Classes

The extension `strandtools` provides two classes to handle the multiview data. `strandtools.SingleViewData` stores the data (images and camera parameters) of a single view, and `strandtools.MultiViewData` stores the multiple single view data.

These classes are implemented in C++, so it is a bit complex to see the details from the code. You can see the overview of the classes by running:

```python
import strandtools

help(strandtools.SingleViewData)
help(strandtools.MultiViewData)
```

## Read/Write

`utils/view.py` provides the utility functions to read/write the multiview data from/to a folder.

```python
import strandtools 
from utils import read_multiview
from utils import read_singleview, write_singleview

# Read multiview data
path = 'path/to/multiview/mydata'
multiviewdata = read_multiview(path) # strandtools.MultiViewData

num = len(multiviewdata)

# Get the first view
view = multiviewdata[0] # strandtools.SingleViewData

# Access the data of the SingleViewData
width, height = view.size()
img_intensity = view.img_intensity
img_depth = view.img_depth 
img_direction = view.img_direction 
camera = view.camera # strandtools.Camera
K = camera.K
R = camera.R
t = camera.t

# Edit the data and write it to a new folder
view.img_intensity = img_intensity * 2.0
write_singleview('path/to/multiview/mydata_new/00', view)
```

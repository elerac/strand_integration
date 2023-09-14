# C++ extension: `strandtools`

This extension provides an actual implementation of LPMVS and a data structure to store multi-view images.

## Installation

### From source

You can install from source. OpenCV and OpenMP are required to build the extension.

```bash
cd cpp_ext
pip install .
```

We use [nanobind](https://github.com/wjakob/nanobind) to implement the C++ extension. You should read nanobind documentation for the details about a build system. 

<!-- ### From wheel

You can also install from wheel. We provide pre-built wheels for Linux and Windows. See `dist` directory and select the appropriate wheel for your environment.

```bash
pip install <wheel file>
```

ToDo: Build wheels for macOS with GitHub Actions. -->

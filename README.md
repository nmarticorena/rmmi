# RMMI:Reactive Mobile Manipulation using an Implicit Neural Map

## Repo organisation

### Deps
The majority of the dependencies can be installed either in pip or conda, we instead follow a submodule approach that includes the following advantages:
- Our forks of the libraries include some small extra functionalities and an update on the building backend
- By changing the building backend we can install the libraries in editable mode using .pixi (setuptools is not compatible) allowing for easier tweeking of the underlying dependencies.

If you need to update any of the deps to re-trigger the compilation of the c/c++ extensions, you need to run:`pixi reinstall [library-name]` such as `pixi reinstall swift`

The dependencies are:
|Name | description|
| neural_robot | extensions of the robotics toolbox in python to expose some of the functionalities as torch.tensors |
| nerf_tools | set of scripts to manage NeRF like dataset and recording|
| isdf | neural sdf used in this work, this can be interchange with newer methods, check [sdf.py](mm_neo/sdf.py) for details on how to add your favorite neural SDF approach|

## Getting Started

To ensure all dependencies are installed in a reproducible manner, we use the package management tool [**pixi**](https://pixi.sh/latest/). If you haven't installed [**pixi**](https://pixi.sh/latest/) yet, please run the following command in your terminal:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```
*After installation, restart your terminal or source your shell for the changes to take effect*. For more details, refer to the [**pixi documentation**](https://pixi.sh/latest/).

Clone the repository and navigate to the project directory:
```bash
git clone --recursive git@github.com:nmarticorena/rmmi.git && cd rmmi
pixi install # Install all the dependencies in a local project
```

## Citation
If you're using RMMI in your research, please cite.

@article{Marticorena2025RMMI,
  author    = {Marticorena, Nicolas and Fischer, Tobias and Haviland, Jesse and Suenderhauf, Niko},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems},
  title     = {{RMMI: Reactive Mobile Manipulation using an Implicit Neural Map}},
  year      = {2025},
}

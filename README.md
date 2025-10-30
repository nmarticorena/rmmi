# Reactive Mobile Manipulation using an Implicit Neural Map

## Repo organisation

### Deps
The majority of the dependencies can be installed either in pip or conda, we instead follow a submodule approach that includes the following advantages:
- Our forks of the libraries include some small extra functionalities and an update on the building backend
- By changing the building backend we can install the libraries in editable mode using .pixi (setuptools is not compatible) allowing for easier tweeking of the underlying dependencies.

If you need to update any of the deps to re-trigger the compilation of the c/c++ extensions, you need to run:`pixi reinstall [library-name]` such as `pixi reinstall swift`

## Install instructions

```bash
git clone --recursive git@github.com:nmarticorena/rmmi.git
```

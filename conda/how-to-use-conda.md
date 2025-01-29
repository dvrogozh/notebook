# Overview

Conda is open source package and environment mamanger. Conda can be installed by few different installers:

* [Miniconda] is a minimal installer provided by Anaconda (company)
* [Anaconda] is a full feature installer which comes with a suite of packages, GUI, etc.
* [Miniforge] is a minimal installer supported by community

If you prefer minimal environments and install only those packages you need - consider using Miniconda or Miniforge. Such projects as PyTorch use Miniforge in their ci.

# Installation

To [install on Linux] download and run one of the Conda installers. For [Miniforge]:
```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
chmod a+x Miniforge3-Linux-x86_64.sh
./Miniforge3-Linux-x86_64.sh
```

By default Conda will be installed to `$HOME/miniforge3`.

You will need to agree to license agreement. You also will be asked to update your shell profile (`~/.bashrc`) to have Conda in your environment. Note that by doing so will activate base conda environment and you will start using application and libraries from it instead of system ones if both versions are installed. This specifically concerns `pip` and `python` even in minimal Conda setup. After logging into a new shell you will enter base Conda environment, so packages installed with `pip` will get installed into this environment instead of `~/.local` as would happen using system installation of `pip`.

Base Conda environment can be activated/deactivated same as any other Conda environment you will further create. Use name `base`:

* To deactivate base environment:
```
conda deactivate
```

* To activate base environment:
```
conda activate base
```

# Working with environments

To create environment:
```
conda create -y -n my-environment
```

To create environment with specific Python version:
```
conda create -y -n my-environment python=3.12
```

To activate:
```
conda activate my-environment
```

To deactivate active environment:
```
conda deactivate
```

To uninstall all package in the environment and destroy the environment:
```
conda remove -n my-environment --all
```

# Using `pip` in Conda environment

Behavior of `pip` depends on which version of `pip` is being used. When Conda environment is activated you get `PATH` configured to pick application installed in the environment. Thus, if `pip` was was not installed in the environment, then system version will be used. In such a case packages installed by `pip` will be installed into `~/.local` - default location for system installation of `pip`. To make sure that `pip` installs packages into activated Conda environment, install `pip` with Conda within this environment, i.e.:
```
conda activate my-environment
conda install pip
```

Note that base Conda environment comes pre-configured with Conda installation and contains few applications such as `python` and `pip`. So, packages installed with `pip` with activated base environment will be installed into Conda base environment

[Miniconda]: https://docs.anaconda.com/miniconda/
[Anaconda]: https://www.anaconda.com/download
[Miniforge]: https://conda-forge.org/download

[install on Linux]: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

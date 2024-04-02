# learn-jax

## Install dependancies
 run the following command:

 to install conda on a TPU VM
 ```shell
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

export PATH="$HOME/miniconda3/bin:$PATH"
source ~/.bashrc
```

To install the conda env and install the dependancies
```shell
 conda create -n jax-llm python=3.10
 conda activate jax-llm
 pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
 ```

 This will install JAX and other required packages to run the tutorial
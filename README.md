## Unsupervised pre-training in Atari ##

<img src="images/vizdoom.gif" width="351">

This is a fork from github.com/pathak22/noreward-rl ([ICML 2017 paper on curiosity-driven exploration for reinforcement learning](http://pathak22.github.io/noreward-rl/)), extending the curiosity-driven approach with perceptual pre-training on a range of Atari environments.

### 1) Installation and Usage
1.  This code is based on [TensorFlow](https://www.tensorflow.org/). To install, run these commands:
  ```Shell
  # you might not need many of these, e.g., fceux is only for mario
  sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb \
  libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig python3-dev \
  python3-venv make golang libjpeg-turbo8-dev gcc wget unzip git fceux virtualenv \
  tmux

  # install the code
  git clone -b master --single-branch https://github.com/pathak22/noreward-rl.git
  cd noreward-rl/
  conda env create -f environment.yml
  conda activate curiosity
  pip install -r src/requirements.txt

  # download models
  bash models/download_models.sh
  ```

2. Running demo
  ```Shell
  cd noreward-rl/src/
  python demo.py --env-id SuperMarioBros-1-1-v0 --ckpt ../models/mario/mario_ICM
  ```

3. Training code
  ```Shell
  cd noreward-rl/src/
  # For Doom: doom or doomSparse or doomVerySparse
  python train.py --default --env-id doom

  # For Mario, change src/constants.py as follows:
  # PREDICTION_BETA = 0.2
  # ENTROPY_BETA = 0.0005
  python train.py --default --env-id mario --noReward

  xvfb-run -s "-screen 0 1400x900x24" bash  # only for remote desktops
  # useful xvfb link: http://stackoverflow.com/a/30336424
  python inference.py --default --env-id doom --record
  ```

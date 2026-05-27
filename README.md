# Deep Lagrangian Networks
The open-source implementation of Deep Lagrangian Networks presented in
- [Lutter and Peters, (2021). Combining Physics and Deep Learning to learn Continuous-Time Dynamics Models, 
arXiv preprint arXiv:2110.01894](https://arxiv.org/abs/2110.01894).
 
- [Lutter et. al., (2019). Deep Lagrangian Networks: Using Physics as Model Prior for Deep Learning, 
International Conference on Learning Representations (ICLR)](https://arxiv.org/abs/1907.04490).

- [Lutter et. al., (2019). Deep Lagrangian Networks for end-to-end learning of energy-based control for under-actuated systems,
 International Conference on Intelligent Robots & Systems (IROS)](https://arxiv.org/abs/1907.04489).

**Installation:** \
For installation this python package can be cloned and installed via pip

It is suggested to work within a venv
```powershell
# Other dependencies (Using JAX CPU version. Optionally, change this to the GPU version)
pip install -U jax
pip install -U dm-haiku
pip install optax
pip install PyQt5

#Use pytorch on GPU
#   To find compatible version, run
#     powershell:
#       nvidia-smi.exe
#     bash:
#       nvidia-smi
#   In the output, look for "CUDA Version"
#   Then get the install command from
#   https://pytorch.org/
#   e.g.
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Or use the requirements file
```powershell
cd deep_lagrangian_networks
python -m pip install -r requirements.txt
```

**Quick Start:** \
Train model and run
```powershell
#rayleigh dissipation, pytorch version
python ./example_DeLaN.py  -m 1
#stribeck formulation, pytorch version 
python ./delan_train_formula.py -m 1
```

Load model and run (requires training first)
```powershell
# pytorch version
python ./example_DeLaN.py -l 1
```

Single trajectory evaluation:
```powershell
# stribeck formulation
python ./eval_traj_csv.py --csv {path_to_csv}
# rayleigh dissipation 
python ./predict_s.py --csv {path_to_csv}
```

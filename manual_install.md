i have modified the setup code to only run on my machines hardware as otherwise it does not build the correct wheel

we want to disable build isolation as my locally installed pytorch is not the standard one but the cuda 12.6 one despite my cuda driver being 12.8

as we are manually building we also need to manually set the package version in setup.py!!!
```bash
python -m pip wheel . --no-deps --no-build-isolation -w ./dist -v
```

after this we install by:
```bash
pip install ./dist/mamba_ssm-*.whl
```

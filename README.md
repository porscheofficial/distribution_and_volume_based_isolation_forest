# Reproduction instructions

This directory contains the code required to reproduce the results presented in the submission "Distribution and volume based scoring for Isolation Forests".

## Setup

The benchmarks are run using the [_**ADBench**: Anomaly Detection Benchmark_](https://arxiv.org/abs/2206.09426). Hence users are required to set up that benchmark. However, since that benchmark is quite large and we don't require all the packages required in that repository, we ask users to follow the following custom setup instructions

1. In a new folder, check out the [ADBench repository](https://github.com/Minqi824/ADBench) at the NeurIPS2022 commit. Assuming you're using `git > 2.5`:
    - `git init`
    - `git remote add origin https://github.com/Minqi824/ADBench`
    - `git fetch origin 6345a6b35d66b460bd5a590f6db9774e59e71487` (downloads ~2GB)
    - `git reset --hard FETCH_HEAD`

3. From this directory, clone our repo and replace one of the ADBench repo files:
    - `git clone https://github.com/porscheofficial/distribution_and_volume_based_isolation_forest.git`
    - `cd distribution_and_volume_based_isolation_forest`
    - `mv data_generator.py ../`

5. Create a virtual environment (we used Python version 3.9.6) and install the requirements from the `requirements.txt` contained in this folder:
    - `pip3.9 install virtualenv`
    - `python3.9 -m virtualenv venv`
    - `source venv/bin/activate`
    - `pip install -r requirements.txt`
  
You can then execute the two notebooks required for generating results and plots from the paper as usual.

## How to cite

Please consider citing our paper if you use our code in your project.

XXX UPDATE when uploaded

```
@bibtex_ref{}
```

## Contributing

This GNN-based oracle factory is openly developed in the wild and contributions (both internal and external) are highly appreciated.
See [CONTRIBUTING.md](./CONTRIBUTING.md) on how to get started.

If you have feedback or want to propose a new feature, please [open an issue](https://github.com/porscheofficial/porscheofficial.github.io/issues).
Thank you! 😊

## Acknowledgements

This project is part of the AI research of [Porsche Digital](https://www.porsche.digital/). ✨


## License

Copyright © 2023 Porsche Digital GmbH

Porsche Digital GmbH publishes this open source software and accompanied documentation (if any) subject to the terms of the [MIT license](./LICENSE.md). All rights not explicitly granted to you under the MIT license remain the sole and exclusive property of Porsche Digital GmbH.

Apart from the software and documentation described above, the texts, images, graphics, animations, video and audio files as well as all other contents on this website are subject to the legal provisions of copyright law and, where applicable, other intellectual property rights. The aforementioned proprietary content of this website may not be duplicated, distributed, reproduced, made publicly accessible or otherwise used without the prior consent of the right holder.

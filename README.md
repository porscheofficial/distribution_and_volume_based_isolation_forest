**This branch contains code of the project that didn't make it into the final project. In particular it contains a branch and bound implementation and a mixed integer nonlinear programming approach to implementing an exact HAS_RARE_PATTERN subroutine that provides an alternative to the isolation forest random forest procedure. This turned out to be performing poorly, we leave it here for the interested reader to build on.**

**Moreover, the requirements from the requirements.txt might not suffice to make the code on this branch work**

# Reproduction instructions

This directory contains the code required to reproduce the results presented in the submission "Distribution and volume based scoring for Isolation Forests".

## Setup

The benchmarks are run using the [_**ADBench**: Anomaly Detection Benchmark_](https://arxiv.org/abs/2206.09426). Hence users are required to set up that benchmark. However, since that benchmark is quite large and we don't require all the packages required in that repository, we ask users to follow the following custom setup instructions

1. Clone the [Github repository](https://github.com/Minqi824/ADBench)
2. Copy this folder into the root directory of the cloned ADBench repo
3. Create a virtual environment (we used Python version 3.9.6) and install the requirements from the `requirements.txt` contained in this folder.

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

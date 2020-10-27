Fabio D'Isidoro - 08.08.2017 - ETH Z�rich

A CUDA-accelerated C++ library for DRR generation using an improved version of the Siddon algorithm.

The library defines a class object SiddonGPu, that loads a CT scan onto the GPU memory when initialized.
A function member of the class  (generateDRR) can be used to generate DRRs.

--------
Modified by Pengyi Zhang for infection-aware DRR generator @2020-08
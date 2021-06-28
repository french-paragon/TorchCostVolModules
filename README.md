# TorchCostVolModules

This repository contain a series of cost volume construction modules for pytorch implemented in pure python without loop or indexing. Coding this was more for the challenge of it than for any practical project. The layers are extremely fast but consume a lot of memory, especially if you try to backpropagate through the actual layer. The modules themselves should work with pytorch only but for the purposes of using np.newaxis (which is much clearer than indexing with None), there is a numpy dependency.

The test code also depend on argparse, matplotlib, time and [the python module provided by my exr-tools suite](https://github.com/french-paragon/exr-tools).

Note that this project is just a personal project and in no other way related to the official pytorch distribution.

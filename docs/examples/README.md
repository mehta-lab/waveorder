`waveorder` is undergoing a significant refactor, and this `examples/` folder serves as a good place to understand the current state of the repository.

Most examples require `pip install waveorder[all]` for `napari` and `jupyter`. Visit the [napari installation guide](https://napari.org/dev/tutorials/fundamentals/installation.html) if napari installation fails.

| Folder      | Requires                   | Description                                                                                           |
|------------------|----------------------------|-------------------------------------------------------------------------------------------------------|
| `configs/`       | `pip install waveorder[all]`           | Demonstrates `waveorder`'s config-file-based command-line interface. |
| `models/`        | `pip install waveorder[all]`      | Demonstrates the latest functionality of `waveorder` through simulations and reconstructions using various models. |
| `maintenance/`   | `pip install waveorder`                | Examples of computational imaging methods enabled by functionality of waveorder; scripts are maintained with automated tests.               |
| `visuals/`       | `pip install waveorder[all]`      | Visualizations of transfer functions and Green's tensors.                                    |
| `deprecated/` | `pip install waveorder[all]`, complete datasets | Provides examples of real-data reconstructions; serves as documentation and is not actively maintained. |
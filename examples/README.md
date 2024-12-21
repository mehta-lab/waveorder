`waveorder` is undergoing a significant refactor, and this `examples/` folder serves as a good place to understand the current state of the repository.

Some examples require `pip install waveorder[examples]` for `napari` and `jupyter`. Visit the [napari installation guide](https://napari.org/dev/tutorials/fundamentals/installation.html) if napari installation fails.

| Folder      | Requires                   | Description                                                                                           |
|------------------|----------------------------|-------------------------------------------------------------------------------------------------------|
| `models/`        | `pip install waveorder[examples]`      | Demonstrates the latest functionality of `waveorder` through simulations and reconstructions using various models. |
| `maintenance/`   | `pip install waveorder`                | Showcases functionality planned for `models/` folder; scripts are maintained with automated tests.               |
| `visuals/`       | `pip install waveorder[examples]`      | Visualizations of transfer functions and Green's tensors.                                    |
| `documentation/` | `pip install waveorder`, complete datasets | Provides examples of real-data reconstructions; serves as documentation and is not actively maintained. |

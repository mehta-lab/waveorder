`waveorder` is undergoing a significant refactor, and this `examples/` folder serves as a good place to understand the current state of the repository.

Some examples require `napari`. Start with `pip install napari[all]`, and visit the [napari installation guide](https://napari.org/dev/tutorials/fundamentals/installation.html) if that fails.  

| Folder      | Requires                   | Description                                                                                           |
|------------------|----------------------------|-------------------------------------------------------------------------------------------------------|
| `models/`        | `waveorder`, `napari`      | Demonstrates the latest functionality of `waveorder` through simulations and reconstructions using various models. |
| `maintenance/`   | `waveorder`                | Showcases functionality planned for `models/` folder; scripts are maintained with automated tests.               |
| `visuals/`       | `waveorder`, `napari`      | Visualizations of transfer functions and Green's tensors.                                    |
| `documentation/` | `waveorder`, complete datasets | Provides examples of real-data reconstructions; serves as documentation and is not actively maintained. |

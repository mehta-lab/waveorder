`waveorder` is undergoing a significant refactor, and this `examples/` folder serves as a good place to understand the current state of the repository.

The `models/` folder demonstrates the latest functionality of `waveorder`. These scripts will run as is in an environment with `waveorder` and `napari` installed. Each script demonstrates a simulation and reconstruction with a **model**: a specific set of assumptions about the sample and the data being acquired. 

The `maintenance/` folder demonstrates the functionality of `waveorder` that we plan to move to `models/`. These scripts can be run as is, and they are being maintained with tests.

The `documentation/` folder consists of examples that demonstrate reconstruction with real data. These examples require access to the complete datasets, so they are not being actively maintained and serve primarily as documentation.
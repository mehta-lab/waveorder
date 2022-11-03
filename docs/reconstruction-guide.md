# Automating reconstructions

`recOrder` is undergoing changes to the way that it handles automated reconstructions.

`recOrder==0.2.0` had an offline mode with `config.yml` files that were used to configure reconstructions. These config files could be generated and saved via the GUI, and the reconstructions could be run via GUI or CLI. 

Although the offline mode had many valuable and convenient features, we found that it had diverged from the online mode and it was difficult to recreate results between online and offline modes. These design limitations led us to the following plan for our upcoming releases:

`recOrder==0.3.0` (release candidate in the coming weeks) will remove the offline mode and use a set of scripts instead. We recommend modifying the reconstruction scripts in `recOrder/examples/` to recreate and automate reconstructions. 

Although these scripts are not as user friendly as a GUI+CLI solution, we are preparing for a much cleaner solution in `1.0.0`, and we appreciate your patience as we go through this change. We ask users who are having any difficulty with the scripts to [open an issue](https://github.com/mehta-lab/recOrder/issues/new/choose) or [send us an email](mailto:shalin.mehta@czbiohub.org,talon.chandler@czbiohub.org).

`recOrder==1.0.0` will use a single mode to acquire and reconstruct the data. We are currently planning a refactor that will enable an "acquire once, quickly iterate your reconstruction" workflow, an "acquire now, reconstruct later" workflow, and a "live acquisition, live reconstruction" workflow, among others.  

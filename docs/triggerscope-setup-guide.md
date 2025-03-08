## Triggerscope Setup for Meadowlark LCs

## What is the Triggerscope?
The [Triggerscope](https://advancedresearch-consulting.com/product/triggerscope-4/) is a device for hardware control and synchronization of cameras, lasers, shutters, galvos, stages and other optoelectronic equipment used popularily used home-built microscopes. This device is compatible with [micromanager](https://micro-manager.org/TriggerScopeMM) and can be used to control the Meadowlark LCs.

![triggerscope](https://sp-ao.shortpixel.ai/client/to_webp,q_glossy,ret_img,w_300,h_225/https://advancedresearch-consulting.com/wp-content/uploads/2022/05/TG4-1.jpg)

# Hardware List

- Meadowlark LCs
- Meadowlark DS5020 
  - Converts the 0-5V signal from the triggerscope to the Meadowlark LCs voltage range.
- [Triggerscope](https://advancedresearch-consulting.com/product/triggerscope-4/)
  - Sends analog 0-5V signal to the Meadowlark DS5020 controlbox to drive the LCs through external analog inputs.

# Setting up the Triggerscope
## Setup with Micromanager
1. Connect the triggerscope via USB
2. Connect the external power supply to the triggerscope
3. Remember to flip on the switch on the back of the triggerscope.
4. Install the MicroManager Triggerscope firmware by following the instructions from your [Triggerscope](https://github.com/micro-manager/TriggerScopeMM) versions.

2. Launch Micromanager, open `Devices > Hardware Configuration Wizard...`, and add the `Triggerscope  Hub` device to your configuration.

3. Confirm your installation by opening Devices > Device Property Browser... and confirming that `Triggerscope DAC` properties appear.

## Setup with Micromanager and Meadowlark LCs

Since we will be driving the LCs with the triggerscope, for this step, we will remove the Meadowlark LCs from the micromanager config file and set the LCs to accept analog input from the triggerscope.

1. Set the LCs to `external mode`. Open the `CellDrive` device and set the `External Mode` property to `True`.
2. Look at the Meadowlark LC control box and find the LCA and LCB connectors.

![meadowlark-connectors]()

3. Connect the LCA and LCB to the Triggerscope DAC output via SMA connectors. Make sure to note which LC (i.e LCA and LCB) is connected to Triggerscope DAC output number. We will use this to setup the triggerscope in Micromanager.

4. Remove the Meadowlark LCs from the micromanager config file.

5. Launch Micromanager with the new micromanager config file.

6. Open the `Triggerscope DAC` device and set the `Output` property to the number of the Triggerscope DAC output that is connected to the LCA and LCB.

7. **TODO**


## Hardware Sequencing

Additionally, the triggerscope can be used as a device that can be sequenced to trigger fast and precisely the optoelectronic hardware in the microscope. In Micromanager, sequencing referes to the pre-computed train of events that will trigger the different pieces of hardware fast and precisly.

To create fast and precise triggering sequences, Micromanager needs to know what devices will be `sequenced` and in what order, typically predetermined by the MDA. The devices that can be sequenced include lightsources, laser combiners, stages, and DACs. Refer to the individual device adapater to check if this devices supports `hardware sequencing`.

## FAQ / Troubleshooting
- The LCs are not changing even if I sweep the voltages on the micromanager properties. 
  - Make sure the LC controller box is connected to the computer via USB
  - Open CellDrive and set the LCs in ["external mode"](#set-the-lcs-to-external-mode)
  - Check the meadowlark LCs are not present in the micromanger config file.
- When I change the triggerscope voltages from the MM device property manager, MM crashes
  - [Check](#plugin-the-triggerscope) that the Triggerscope is connected via USB and connected to its power supply through the barrel connector. 
- 

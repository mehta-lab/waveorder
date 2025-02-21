# Buyer's Guide

## Quantitative phase imaging:

You can use a transmitted light source (LED or a lamp) and a condenser commonly available on almost all microscopes. In addition to the transmitted light imaging path, you will need a motorized stage for acquiring through-focus image stacks. 

## Quantitative polarization imaging (PolScope):

The following list of components assumes that you already have a transmitted light source (LED or a lamp) and a condenser. 

Buyers have two options:
1. buy a complete hardware kit from the OpenPolScope project, or
2. assemble your own kit piece by piece. 

### Buy a kit from the OpenPolScope project
 
- Read about the [OpenPolScope Hardware Kit](https://openpolscope.org/pages/OPS_Hardware.htm).
- Complete the [OpenPolScope information request form](https://openpolscope.org/pages/Info_Request_Form.htm).

### Buy individual components

The components are listed in the order in which they process light. See the build video here to see how to assemble these components on your microscope. 

https://github.com/user-attachments/assets/a0a8bffb-bf81-4401-9ace-3b4955436b57

| Part                     | Approximate Price | Notes                       |
|--------------------------|-------------------|-----------------------------|
| Illumination filter | $200 | We suggest [a Thorlabs CWL = 530 nm, FWHM = 10 nm notch filter](https://www.thorlabs.com/thorproduct.cfm?partnumber=FBH530-10).|
| Circular polarizer | $350 | We suggest [a Thorlabs 532 nm, left-hand circular polarizer](https://www.thorlabs.com/thorproduct.cfm?partnumber=CP1L532).|
| Liquid crystal compensator | $6,000 | Meadowlark optics LVR-42x52mm-VIS-ASSY or LVR-50x60mm-VIS-POL-ASSY. Although near-variants are listed in the [Meadowlowlark catalog](https://www.meadowlark.com/product/liquid-crystal-variable-retarder/), this is a custom part with two liquid crystals in a custom housing. [Contact Meadowlark](https://www.meadowlark.com/contact-us/) for a quote.|
| Liquid crystal control electronics | $2,000 | [Meadowlark optics D5020-20V](https://www.meadowlark.com/product/liquid-crystal-digital-interface-controller/). Choose the high-voltage 20V version. 
| Liquid crystal adapter | $25-$500 | A 3D printed part that aligns the liquid crystal compensator in a microscope stand's illumination path. Check for your stand among the [OpenPolScope `.stl` files](https://github.com/amitabhverma/Microscope-LC-adapters/tree/main/stl_files) or [contact us](compmicro@czbiohub.org) for more options.|
| Circular analyzer (opposite handedness) | $350 | We suggest [a Thorlabs 532 nm, right-hand circular polarizer](https://www.thorlabs.com/thorproduct.cfm?partnumber=CP1R532).|

If you need help selecting or assembling the components, please start an issue on this GitHub repository or contact us at compmicro@czbiohub.org.

## Quantitative phase and polarization imaging (QLIPP):

Combining the Z-stage and the PolScope components listed above enables joint phase and polarization imaging with `recOrder`.
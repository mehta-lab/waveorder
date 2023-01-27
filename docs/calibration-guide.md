# Calibration guide
This guide describes `recOrder`'s calibration routine with details about its goals, parameters, and evaluation metrics. 

## Why calibrate?

`recOrder` sends commands via MicroManager (or a TriggerScope) to apply voltages to the liquid crystals which modify the polarization of the light that illuminates the sample. `recOrder` could apply a fixed set of voltages so the user would never have to worry about these details, but this approach leads to extremely poor performance because

- the sample, the sample holder, lenses, dichroics, and other optical elements introduce small changes in polarization, and 
- the liquid crystals' voltage response drifts over time.

Therefore, recalibrating the liquid crystals regularly (definitely between imaging sessions, often between different samples) is essential for acquiring optimal images. 

## Finding the extinction state

Every calibration starts with a routine that finds the "extinction state": the polarization state (and corresponding voltages) that minimize the intensity that reaches the camera. If the analyzer is a right-hand-circular polarizer, then the extinction state is the set of voltages that correspond to left-hand-circular light in the sample. 

## Setting a goal for the remaining states: swing 

After finding the (circular) extinction state, the calibration routine finds the remaining states. The "swing" parameter is best understood 
sets the target ellipticity of these states in fractions of a wave. For example, a swing of 0 

## Evaluating a calibration: extinction



## Calculating extinction from measured intensities (advanced topic)

$\begin{equation}I_{\text{ellip}}(\chi) = I_{\text{mod}}\sin^2(\pi\chi) + I_{\text{ext}}\end{equation}$

$\begin{equation}I_{\text{ext}} = I_{\text{leak}} + I_{\text{bl}}\end{equation}$

$\begin{equation}\text{Extinction} \equiv \frac{I_{\text{mod}} + I_{\text{leak}}}{I_{\text{leak}}}\end{equation}$

$\begin{equation}\text{Extinction} = \frac{1}{\sin^2(\pi\chi)}\frac{I_{\text{ellip}}(\chi) - I_{\text{ext}}}{I_{\text{ext}} - I_{\text{bl}}} + 1\end{equation}$

## Summary: step-by-step calibration procedure
1. Temporarily close the shutter to measure the black level. 
2. Find the extinction state by finding voltages that minimize the intensity that reaches the camera. 
3. Use the swing value to immediately find the first elliptical state, and record the intensity on the camera. 
4. For each remaining elliptical state, keep the polarization orientation fixed and optimize the voltages to match the intensity of the first elliptical state. 
5. Store the voltages and calculate the extinction value. 
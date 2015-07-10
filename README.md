# Hydrus wrapper
Wrapper to automate local sensitivity analysis of a hydrus implemented model 

Implemented to support the Phd of Meisam Rezaei, dealing with improved irrigation strategies. Hydrus is used to model the soil water content of the soil profile in time (1D simulation).

The loose set of functions automates the workflow by changing the parameter values in the input-files of Hydrus, running the .exe and extracting the modelled outcome:

* link with the Hydrus .exe file to run the model
* changes of parameter handling
* numerical approach for local sensitivity analysis
* visualisation of the results


Publication:
http://www.researchgate.net/publication/260157131_Optimizing_Hydrus_1D_for_irrigation_management_purposes_in_sandy_grassland

and

Journal: HESS
Title: Sensitivity of water stress in a two-layered sandy grassland soil to variations in groundwater depth and soil hydraulic parameters
Author(s): M. Rezaei, P. Seuntjens, I. Joris, W. Boënne, S. Van Hoey4, P. Campling, and W. M. Cornelis 

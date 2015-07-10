# -*- coding: utf-8 -*-
"""
Parameter Response visualisation

Project: Phd Meisam Rezaei
Author: Van Hoey Stijn
"""

#------------------------------------------------------------------------------
#Import necessary modules and packages-----------------------------------------
from Meisam_definitions_file import *
import os
import datetime
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
#end impor---------------------------------------------------------------------
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#   INFORMATION
#------------------------------------------------------------------------------
#Layer 1: [0.4, 0.015, 2.4, 2.18, 0.5];
#layer 2: [0.35, 0.01965, 2.5, 2.271, 0.5)
#Range of parameters:
#thetas, m3 m−3 Saturated water content                0.2–0.7
#alfaVG, m−1 Mualem–van Genuchten shape factor       0.001–20.0
#nVG Mualem–van Genuchten shape factor            1.1–9.0
#Ks, m s−1 Saturated hydraulic conductivity       0.1-25 cm/h
#l Pore-connectivity parameter                    0.1–1. 0
#END INFORMATION---------------------------------------------------------------
#------------------------------------------------------------------------------
    
    
#------------------------------------------------------------------------------
#   TUTORIAL HOW TO USE THE MODEL RUNNING
#------------------------------------------------------------------------------
###Put the default parameters
#path_to_model='D:\\Python_sensitivity\\1Dmodel2'
##create_default_selector(path_to_model)
###replace a parameter
#replaceInputWater(path_to_model, 3., parname='Ks', layer=1)
###run the model
#runHydrus(guessed_runtime=4, path_to_dir= 'D:\\Python_sensitivity\\1Dmodel2',install_dir="C:\\Program Files (x86)\\PC-Progress\\Hydrus-1D 4.xx")
####prepare the filtered output file
#filter_on_timestep(infile='1Dmodel2\Obs_Node.out',outfile='1Dmodel2\Obs_Node_filtered.out', nnodes=4)
####read output in dataframe
#df = readoutput_to_dataframe(filename='1Dmodel2\Obs_Node_filtered.out', startdate='3/1/2012 00:00', enddate='6/13/2012 03:00', variable='theta', nnodes=4)
####plot the outputs in graph
##df.plot(subplots=True, figsize=(16, 8), yticks=[0.0,0.2,0.4]) 

##get the rain from the data
#rain = pd.read_csv('D:\\Python_sensitivity\\1Dmodel2\\rain.csv', index_col=0, names=['rain'], parse_dates=True, dayfirst=True)
#quickplot(df,nnodes=4)              

#------------------------------------------------------------------------------
#   TUTORIAL SENSITIVTY
#------------------------------------------------------------------------------
#path_to_model='D:\\Python_sensitivity\\1Dmodel2'
#CAS, CPRS, CTRS, average_out, df_par_plus, df_par_min = calculate_sens(path_to_model, 22.2, perturbation_factor = 0.1, parameter_name='Ks', parameter_layer=1,
#                   startdate='5/4/2011 13:00', enddate='9/2/2011 11:00', variable = 'theta', guessed_runtime=8,
#                    install_dir="C:\\Program Files (x86)\\PC-Progress\\Hydrus-1D 4.xx")                   
                              
####DO SENSITIVITY FOR ALL
##par_names=['ths','Alfa','n','Ks','l']
#par_names=['Ks', 'Alfa', 'n', 'ths', 'l']
##par_names=['Ks']
##par_values=([2.18],[2.271])     
#par_values=([2.18, 0.015, 2.4, 0.4, 0.5],[2.271, 0.0196, 2.271, 0.35, 0.5])     
#local_sensitivity(path_to_model,par_names, par_values, nnodes=4, guessed_runtime=2, 
#                  startdate='3/1/2012 00:00', 
#                  enddate='6/13/2012 03:00',
#                  install_dir="C:\\Program Files (x86)\\PC-Progress\\Hydrus-1D 4.xx") 
#
#plot_sensitivity(par='Ks', senstype="CTRS", nnodes=4)
#plot_sensitivity(par='Alfa', senstype="CTRS", nnodes=4)
#plot_sensitivity(par='n', senstype="CTRS", nnodes=4)
#plot_sensitivity(par='ths', senstype="CTRS", nnodes=4)
#plot_sensitivity(par='l', senstype="CTRS", nnodes=4)
#
#plot_sensitivity(par='Ks', senstype="CPRS", nnodes=4)
#plot_sensitivity(par='Alfa', senstype="CPRS", nnodes=4)
#plot_sensitivity(par='n', senstype="CPRS", nnodes=4)
#plot_sensitivity(par='ths', senstype="CPRS", nnodes=4)
#plot_sensitivity(par='l', senstype="CPRS", nnodes=4)
    
    
#------------------------------------------------------------------------------
#   PARAMETER RESPONSE SURFACE
#------------------------------------------------------------------------------

allpars=['ths','Alfa','n','Ks','l']
allpars_min=[.2,0.001,1.1,0.1,0.1]
allpars_max=[0.7,20.,9.,25,1.]

par1 = 'Ks'
par2 = 'n'
#n is third par
par1_layer = 1
par2_layer = 1


#Read the measurements and set the start-end date to evaluate
meas = read_meas(filename='vocht_metingen.csv', start='27/6/2011 20:00', end='9/2/2011 11:00')
#set the path where the model inputs are placed
path_to_model= "D:\\Projecten\\2013_Meisam\\Par_response\\2013_5_1dmodel"

#Perform the response surface for two parameters:------------------------------
parspace, ax1 = par_response_surface(path_to_model, 
                                          allpars_min[allpars.index(par1)],  allpars_max[allpars.index(par1)], 
                                          allpars_min[allpars.index(par2)], allpars_max[allpars.index(par2)], 
                                          par1, par2, meas, par1_layer = par1_layer, par2_layer = par2_layer, 
                                          ndx=10, ndy=10, startdate='5/14/2011 08:00', enddate='9/2/2011 11:00',
                                         meas_start = '5/14/2011 08:00', meas_end = '9/2/2011 11:00',
                                         saveit=True, plotnlines= 7, install_dir="C:\\Program Files\\PC-Progress\\Hydrus-1D 4.xx")
                                         #ADAPT INSTALL DIR!!!
ax1.set_xlabel(r'$K_s$', fontsize=16)
ax1.set_ylabel(r'$\alpha$', fontsize=16)


##If already done, only get output from file and plot it----------------------------------
#ax2 = load_parspace("parspace_Ks_Alfa_2013-05-29.txt", allpars_min[allpars.index(par1)],  
#                      allpars_max[allpars.index(par1)], 
#                      allpars_min[allpars.index(par2)], 
#                      allpars_max[allpars.index(par2)], plotnlines= 2)
#
#ax2.set_xlabel(r'$K_s$', fontsize=16)
#ax2.set_ylabel(r'$\alpha$', fontsize=16)


#If already done, only plot it----------------------------------
#parspace_w = parspace.copy()
parspace_w[np.where(parspace > 1e7)] = -0.99
#ax2 = plot_parspace(parspace, allpars_min[allpars.index(par1)],  
#                      allpars_max[allpars.index(par1)], 
#                      allpars_min[allpars.index(par2)], 
#                      allpars_max[allpars.index(par2)], plotnlines= 1,
#                      interpol = 'nearest')
#
#ax2.set_xlabel(r'$K_s$', fontsize=16)
#ax2.set_ylabel(r'$\alpha$', fontsize=16)
#ax2.set_title('White area is giving numerical convergence issues')
#plt.savefig('convergence_problems_'+par1+'_'+par2+'_layer'+str(par1_layer)+str(par2_layer)+'.png')

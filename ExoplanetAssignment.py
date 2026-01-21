# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 16:23:32 2024

@author: Sam
"""


# Exoplanet Assignment

# -----------------------------------------------------------------------------

# name: sam_lawrence

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette('colorblind') ; palette = sns.color_palette()

# constants

g = 6.67430e-11  # gravitational constant [m^3 kg^-1 s^-2]

m_sun = 1.99e30  # solar mass [kg]

m_jup = 1.9e27  # jupiter mass [kg]

# my assigned values

m_star = 1.1 * m_sun  # stellar mass

m_planet = 0.5 * m_jup  # planet mass

eccentricity = 0.35  # eccentricity

orbital_period = 3.19 * 24 * 3600  # orbital period in seconds

inclination = np.pi / 2  # 90 degree inclination [radians]

# calculated parameters

#  semi-amplitude equation from slides

k = ((m_planet * np.sin(inclination)) / ((m_star + m_planet) ** (2/3)) ) *\
    ((2 * np.pi * g) / orbital_period) ** (1/3) * (1 / np.sqrt(1 - eccentricity ** 2))
    
# arguments of periastron(time-independent angle between line of nodes and periastron)
omega_values = np.linspace(0.0, 2 * np.pi, 7)

# time series [state how zero point is set]
time = np.linspace(0, 2 * orbital_period, 1000)


# calculating the true anomaly (time dependant angle between planet and periastron)


def true_anomaly(mean_anomaly, eccentricity): # using newton-raphson method
    
    M = mean_anomaly # abreviating label
    
    e = eccentricity # abreviating label
    
    E = M # initialising to the mean_anomaly from keplers relation of elliptical orbits
    
    while True:
        
        # equation from the hints: E_n = E_n+1 - f(E) / f'(E)
        E_n = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        
        if abs(E_n - E) < 1e-6: # iteratively improving estimate of eccentric_anomaly
            
            break # breaks loop when eccentric_anomaly is within accuracy tolerance
        
        E = E_n # updates eccentric_anomaly estimate

    # true anomaly formula from [Jon Toellner Orbital Dynamics pt.12]
    
    f = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan( E / 2 ))
    
    return f # returns true anomaly



# finding and plotting radial velocity for each argument of periastron


fig, axs = plt.subplots(figsize=(10, 6), gridspec_kw = {'wspace':0, 'hspace':0}, dpi = 300)

for omega in omega_values: # plotting rv for arguments of periastron [time independant]
    
    radial_velocity = []
    
    for t in time: 
        
        # numerically solving for [time dependent] true anomaly and finding radial velocity
        
        mean_anomaly = (2 * np.pi / orbital_period) * t
        
        # mean anomaly ... neglecting phase offset [found in hints]
        
        f = true_anomaly(mean_anomaly, eccentricity)
        # numerically solving for true anomaly
        
        v_r = k * (np.cos(f + omega) + eccentricity * np.cos(omega))
        # finding radial velcity
        
        radial_velocity.append(v_r)
    
    plt.plot(time / (24*3600), radial_velocity, label= f'$\omega$ = {omega:.2f} [rad]',\
             linewidth = 1.1, alpha = 0.9)
        # plotting radial velocity for this argument of periastron

plt.axhline(y = 0, color='k', linestyle='-',\
            linewidth = 0.7, alpha = 0.5)
plt.axhline(y = k, color='k', linestyle='-',label=\
            f'K = {k:.1f} [m/s]', linewidth = 1.1, alpha = 0.9)
plt.axvline(x=orbital_period / (24*3600), color='k', linestyle='--', label=\
            f'P = {orbital_period / (24 * 3600)} [days]', linewidth = 1.1, alpha = 0.9)
plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), fontsize = 11)
plt.suptitle('Radial Velocity Curve', x = 0.5 , y = 0.95, fontsize = 13)
plt.xlabel('Time [days]', labelpad = 10, fontsize = 11)
plt.xlim(0, 2 * orbital_period / (24*3600))
plt.ylabel('Radial Velocity [m/s]', labelpad = 10, fontsize = 11)
plt.text(0.905, 0.7455, 'K', fontsize=13, color='k', transform=plt.gcf().transFigure)
plt.text(0.508, 0.095, 'P', fontsize=13, color='k', transform=plt.gcf().transFigure)
plt.show()

# end of code
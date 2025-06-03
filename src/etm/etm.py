#%%
from etm.TyphoonTrack import TyphoonTrack
import numpy as np
import pyproj
import matplotlib.pyplot as plt

#%%
P_CONSTANT = 1013
RHOA = 1.22
f = 2 * 7.29 * 10 **-5 * np.sin(np.deg2rad(35))

grs80 = pyproj.Geod(ellps='GRS80')

def cal_pressure(center_pressure:float,
                    r0:float=50,
                    r:np.ndarray=None):
    dp = P_CONSTANT - center_pressure
    return center_pressure + dp * np.exp(-r0/r)

def cal_gradient_velocity_top(center_pressure:float,
                                r0:float=50,
                                r:np.ndarray | float = None):
    dp = (P_CONSTANT - center_pressure) * 100
    if isinstance(r, np.ndarray):
        r[r == 0] = 1e-10
    elif isinstance(r, float):
        if r == 0:
            r = 1e-10
    else:
        raise ValueError("r must be a numpy array or a float")

    Ugr = (-0.5*r*f+np.sqrt((r*f *0.5)**2 + dp / RHOA * r0/r * np.exp(-r0/r)))
    return Ugr

def cal_gradient_velocity(center_pressure:float,
                            r0:float=50.,
                            r:np.ndarray | float = None,
                            x:np.ndarray | float = None,
                            y:np.ndarray | float = None,
                            x0:float = 0,
                            y0:float = 0
                        ):
    """
    傾度風の計算
    center_pressure: 中心気圧[hPa]
    r0: 半径[km]
    """
    Ugr = cal_gradient_velocity_top(center_pressure, r0, r)
    # c1 = cal_c1(Ugr, r0, r, x0, y0, x, y)
    c1 = 0.66
    dx = x - x0
    dy = y - y0
    dx[dx == 0] = 1e-10
    ugr_x = c1 * Ugr * np.cos(np.arctan2(dy, dx) + np.deg2rad(90 + 30))
    ugr_y = c1 * Ugr * np.sin(np.arctan2(dy, dx) + np.deg2rad(90 + 30))
    return ugr_x, ugr_y

def cal_c1(urg, r0=50, r=None,x0=0,y0=0,x=0,y=0):
    r0 = 50
    X = r/r0
    Xp = 0.5
    k = 2.5
    c1_inf = 2/3
    c1_XP = 1.2
    c1 = c1_inf + (c1_XP - c1_inf)*(X/Xp)**(k-1.) * np.exp((1.-1/k)*(1.-(X/Xp)**(k)))
    return c1

def cal_utf_velocity(center_pressure,
                        r0:float,
                        r_array:np.ndarray,
                        moving_velocity:float,
                        direction:float
                        ):
    
    ugr_r = cal_gradient_velocity_top(center_pressure,r0,r_array)
    ugr_r0 = cal_gradient_velocity_top(center_pressure,r0,r0)
    utf = ugr_r / ugr_r0 * moving_velocity
    c2 = 0.66
    utf_x = c2 * utf * np.cos(np.deg2rad(direction))
    utf_y = c2 * utf * np.sin(np.deg2rad(direction))
    return utf_x, utf_y


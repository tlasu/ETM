import numpy as np

from etm.TyphoonTrack import geod_inv

P_CONSTANT = 1013
RHOA = 1.22
OMEGA = 7.29e-5  # 地球の自転角速度[rad/s]
DEFAULT_LAT = 35.0


def coriolis_parameter(lat: float | np.ndarray = DEFAULT_LAT):
    """
    コリオリパラメータ f = 2Ωsin(φ) [1/s]
    lat: 緯度[度]
    """
    return 2 * OMEGA * np.sin(np.deg2rad(lat))


def _avoid_zero(r):
    """
    r=0 によるゼロ除算を避ける（入力配列は書き換えない）
    """
    if isinstance(r, np.ndarray):
        return np.where(r == 0, 1e-10, r)
    elif isinstance(r, (int, float)):
        return 1e-10 if r == 0 else float(r)
    raise ValueError("r must be a numpy array or a float")


def cal_pressure(center_pressure:float,
                    r0:float=50,
                    r:np.ndarray=None):
    """
    Myersの気圧分布
    center_pressure: 中心気圧[hPa]
    r0: 最大風速半径[km]
    r: 台風中心からの距離[km]
    """
    dp = P_CONSTANT - center_pressure
    return center_pressure + dp * np.exp(-r0/r)

def cal_gradient_velocity_top(center_pressure:float,
                                r0:float=50,
                                r:np.ndarray | float = None,
                                lat:float = DEFAULT_LAT):
    """
    傾度風速[m/s]
    center_pressure: 中心気圧[hPa]
    r0: 最大風速半径[km]
    r: 台風中心からの距離[km]
    lat: コリオリパラメータを求める緯度[度]
    """
    dp = (P_CONSTANT - center_pressure) * 100  # hPa -> Pa
    r = _avoid_zero(r)
    f = coriolis_parameter(lat)
    r_m = r * 1000  # km -> m（コリオリ項は距離の次元を持つ）

    Ugr = (-0.5*r_m*f+np.sqrt((r_m*f *0.5)**2 + dp / RHOA * r0/r * np.exp(-r0/r)))
    return Ugr

def cal_gradient_velocity(center_pressure:float,
                            r0:float=50.,
                            r:np.ndarray | float = None,
                            x:np.ndarray | float = None,
                            y:np.ndarray | float = None,
                            x0:float = 0,
                            y0:float = 0,
                            lat:float | None = None
                        ):
    """
    傾度風の計算
    center_pressure: 中心気圧[hPa]
    r0: 最大風速半径[km]
    r: 台風中心からの距離[km]
    x, y: 計算点の経度・緯度
    x0, y0: 台風中心の経度・緯度
    lat: コリオリパラメータを求める緯度[度]（省略時は台風中心の緯度 y0）
    """
    if lat is None:
        lat = y0
    Ugr = cal_gradient_velocity_top(center_pressure, r0, r, lat)
    # c1 = cal_c1(Ugr, r0, r, x0, y0, x, y)
    c1 = 0.66
    # 台風中心から計算点への方位角（北=0°、時計回り）を数学角（東=0°、反時計回り）に変換
    azimuth = geod_inv(x0, y0, x, y)[0]
    theta = np.deg2rad(90 - azimuth)
    # 反時計回りの接線方向（+90°）から中心向きに30°吹き込む
    ugr_x = c1 * Ugr * np.cos(theta + np.deg2rad(90 + 30))
    ugr_y = c1 * Ugr * np.sin(theta + np.deg2rad(90 + 30))
    return ugr_x, ugr_y

def cal_c1(urg, r0=50, r=None,x0=0,y0=0,x=0,y=0):
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
                        direction:float,
                        lat:float = DEFAULT_LAT
                        ):
    """
    移動風の計算
    moving_velocity: 台風の移動速度[m/s]
    direction: 進行方向の方位角[度]（北=0°、時計回り。TyphoonTrack.set_directionと同じ定義）
    lat: コリオリパラメータを求める緯度[度]
    """
    ugr_r = cal_gradient_velocity_top(center_pressure,r0,r_array,lat)
    ugr_r0 = cal_gradient_velocity_top(center_pressure,r0,r0,lat)
    utf = ugr_r / ugr_r0 * moving_velocity
    c2 = 0.66
    # 方位角 -> 東西(x)・南北(y)成分
    utf_x = c2 * utf * np.sin(np.deg2rad(direction))
    utf_y = c2 * utf * np.cos(np.deg2rad(direction))
    return utf_x, utf_y

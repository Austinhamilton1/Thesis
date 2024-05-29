SensorReader App

GPS: systime, latitude, longitude, altitude, variance

Inertial: systime, type_of_sensor, x_axis, y_axis, z_axis
type_of_sensor = 1 accelerometer; 2 magnetometer; 3 gyroscope

Wi-Fi: systime, ssid, mac, rssi

systime -> systime_unix = first 10 digits; systime_decimals_sec = digits from 11 onwards
e.g. systime = 1411812293748781273 -> systime_unix = 1411812293; systime_decimals_sec = 0.748781273
     systime = 1411812294182 -> systime_unix = 1411812294; systime_decimals_sec = 0.182
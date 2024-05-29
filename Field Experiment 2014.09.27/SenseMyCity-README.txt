SenseMyCity App

Accelerometer: accelerometer.seconds.Unix, accelerometer.accx.Hex, accelerometer.accy.Hex, accelerometer.accz.Hex
Gyroscope: gyroscope.seconds.Unix, gyroscope.gyrx.Hex, gyroscope.gyry.Hex, gyroscope.gyrz.Hex
Magnetometer: magnetometer.seconds.Unix, magnetometer.magx.Hex, magnetometer.magy.Hex, magnetometer.magz.Hex
- Inertial data in hexadecimal: convert 2 bytes (4 hexadecimal characters) from hex to decimal
IF > 32767 (7FFF) get negative signed value: value-65536 (e.g. FFFF=-1, FFFE=-2, …)
Accelerometer: value/256
Gyroscope: value/512
Magnetometer: value/8

Location: location.seconds.Unix	, location.millis, location.gpstime, location.lat, location.lon, location.alt, location.track, location.speed.M-S, location.acc, location.nsats
- systime: seconds:location.seconds.unix and milliseconds: location.millis

Wfm: wfm.seconds.Unix, wfm.mac_addr.Mac, wfm.snr, wfm.mac_addr.Essid, wfm.mac_addr.Frequency, wfm.mac_addr.Auth

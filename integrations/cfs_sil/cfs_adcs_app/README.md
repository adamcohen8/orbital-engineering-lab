# OEL Reference ADCS cFS App

`oel_adcs_app` is the reference closed-loop flight application for the OEL cFS
SIL mission. It subscribes to `OEL_SIL_SENSOR_MID`, runs a quaternion PD
attitude-hold law, and publishes `OEL_SIL_ACTUATOR_MID`.

This app exists to prove the live Software Bus path and provide a deterministic
reference mission. Customer flight applications replace it while retaining the
same bridge contract or a later mission-defined interface mapping.

Default behavior holds the identity body-to-navigation quaternion, damps body
rates, limits commanded torque to 0.05 N m, and publishes a 500 ms command
validity interval. Tuning lives in `fsw/inc/oel_adcs_app.h`.

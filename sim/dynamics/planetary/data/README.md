# Planetary resource inventory

OEL's planetary environment contract is bring your own coefficients and
atmosphere data. The built-in gravity and atmosphere datasets here are the
Mars reference example. Other bodies share basic constants and rotation
definitions; their gravity and atmosphere datasets are supplied by users.
Mars can also use custom resources through the same import contracts.

These are publicly archived NASA/NAIF science data and explicit OEL processing
definitions. No customer inputs, proprietary atmosphere package, private
provider responses or generated mission report is included. Source URLs and
SHA-256 identities are retained in manifests/JSON definitions. OEL does not
retrieve data at runtime. The original files are preserved, with source archive
labels; retain attribution when redistributing them.

| Files | Source and role |
| --- | --- |
| `gmm2b80.sha`, `gmm2b80.lbl`, `gmm2b_manifest.json` | [MGS MORS 1008 gravity archive](https://pds-geosciences.wustl.edu/mgs/mgs-m-rss-5-sdp-v1/mors_1008/sha/), original `ggm2bc80.sha` and corresponding label; static GMM-2B field, IAU 1991, fully normalized, radius 3397 km. Scientific reference: DOI [10.1029/2000JE001426](https://doi.org/10.1029/2000JE001426). |
| `maven718_profile.tab`, `maven718_profile.xml` | [MAVEN accelerometer P00718 original product](https://atmos.nmsu.edu/PDS/data/PDS4/MAVEN/acc_bundle/l3/2015/02/mvn_acc_l3_pro-acc-p00718_20150212_v02_r01.tab), version 02/revision 01, DOI [10.17189/1518947](https://doi.org/10.17189/1518947). |
| `mars_reference_atmosphere.json` | OEL fixed reference selection: inbound seconds from periapsis <= -14, altitude 135–165 km, 99-second density > 3 times its reported sigma; sort by altitude, convert kg/km³ × 1e-9 to kg/m³. Actual bounds 135.017–164.937 km. 169 rows, no fitting or averaging of different branches. Co-rotating zero winds are an explicit assumption. |
| `gm_de440.tpc`, `body_constants_manifest.json` | [NAIF DE440 GM kernel](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc), physical centers rather than planet-system barycenters. Default body radii use archived IAU 2000 references; nominal solar radius uses IAU 2015 Resolution B3 (695700 km). Harmonic GM/radius override point-mass defaults coherently when selected. |
| `pck00008.tpc`, `*_iau2000_rotation.json` | [Archived NAIF PCK 00008](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/a_old_versions/pck00008.tpc), IAU 2000 pole/meridian and periodic elements. OEL transcribes polynomial and periodic coefficients and converts rates to the native input contract. |
| `mars_iau1991_rotation.json` | Exact old Mars IAU 1991 values documented in [PCK 00005](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/a_old_versions/pck00005.tpc): RA 317.681 - 0.108 T, DEC 52.886 - 0.061 T, W 176.868 + 350.891983 d. The current IAU 1994 block in that kernel has a different W constant and is not used. |

MAVEN geodetic altitude uses a navigation ellipsoid a=3396.19 km,
b=3376.20 km, rather than GMM-2B's harmonic reference radius. The archive SIS
and [Zurek et al. 2017](https://doi.org/10.1002/2016JA023641) describe the
navigation altitude and density conditions. The original XML documents sample
columns and uncertainty; calibration biases and along-track gradients mean
these values do not define a global or seasonal Mars atmosphere.

The rotation JSON coverage 1972–2030 is an implementation bound. Static
archived orientations and the low-order TT–TDB approximation are not current
precision-orientation products. Numerical qualification is described in
[Planetary ONP](../../../../docs/planetary-onp.md).

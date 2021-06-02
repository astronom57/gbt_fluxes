# gbt_fluxes
This script gbt_fluxes.py extracts sources' fluxes from the GBT pointing scans performed before RadioAstron observations. Actual data are provided upon a reasonable request.

The general logic behind the script is: * match GBT observations codes with those of RadioAstron * for each relevant observation, extract the last cross-scans in both polarisations (the last ones should be the best ones) * for each frequency, perform basic cleaning of the data, see details in the description of clean_anyband(). Optionally produce plots * Write out the cleaned data on a per band basis

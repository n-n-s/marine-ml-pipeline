# Data

This directory contains wave buoy data used for training and evaluating the wave height prediction model.

## Data Source

**Real-time coastal monitoring data from the UK National Network of Regional Coastal Monitoring Programmes**

- **Attribution**
    > "Real time data displayed on this page are from the 
      [Regional Coastal Monitoring Programme](https://coastalmonitoring.org), made freely available under the terms of the
      [Open Government Licence](https://coastalmonitoring.org/CCO_OGL.pdf). 
      Please note that these are real-time data and are not quality-controlled."
- **API**: https://coastalmonitoring.org/ccoresources/api/

### Collection Details

- **Collection Period**: January 2025 - December 2025
- **Temporal Resolution**: Half-hourly observations
- **Buoy Locations**: 
  - Penzance (site_id 75)
  - Porthleven: (site_id 107)

## Reproducing the Data Collection

To collect fresh data from the API you may use the 
[coastal-monitoring-client python library](https://github.com/n-n-s/coastal-monitoring-client).

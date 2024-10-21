# SAMXL_grinding_model

## Overview
Python script which create a model to predict parameters in grinding process from grinding test data.
Currently, there are three models:
1. `rpm_correction_model`: input with avg_force, and rpm_setpoint, the model predicts the avg_RPM that would happen during grinding.
2. `volume_model_svr`: input with grind_time, avg_RPM, avg_force, and initial grinding belt wear, the model predicts volume loss from grinding.


## Installation
clone the latest version from this repository into your workspace

```bash
git clone git@github.com:panin-ananwa/grinder_model.git
```

## volume_model_svr.py
load test data (in .csv format), then create and save a volume prediction model, using Support Vector Regression (SVR), according to the set file name and path in the code


## rpm_correction_model.py
load test data (in .csv format), then create and save a rpm correction model, Support Vector Regression (SVR), according to the set file name and path in the code


## grind_settings_generator.py
Python script which test the two models capability to predict the required settings to achieve the desired material volume removal.
With fixed input RPM and belt wear, the script predict the remaining required settings: force and grind_time


### **Meeting Summary - 04/13/2026 - Sai & Erdong (In-person)**

**Virtual Test Environment Setup**
We discussed the setup of the virtual test environment. The test configuration included placeholder values for most processes, while real system measurements were used for four key stages:

-Data extraction from storage
-Preprocessing
-Model data reading
-Model inference time

Test numbers were recorded with intentional pauses to simulate realistic execution timing.

**Use of Decision Log as Historical Input**
An alternative testing approach was explored where the model’s own decision history (decision log) was used as the input history instead of actual ramp data. The idea was to recursively use past predictions to inform future decisions.

**Observations from Initial Experiments**
The results from early experiments were not as expected:

-When using actual data for the first 30 seconds and then switching to the decision log, the model predicted all zeros (“stay”).
-When forcing an increase for the first 10 steps, the model predicted all ones (“increase”) thereafter.
-Even with only 2 initial forced increase steps, the model still converged to predicting all ones.

These behaviors suggest strong bias or instability when the model relies on its own predictions as input.

**Paper Discussion**
We reviewed the paper:
**Assaf & Schumann (2019), “Explainable Deep Neural Networks for Multivariate Time Series Predictions,” IJCAI.**

Key points:

-The model architecture combines 2D CNNs (for extracting time-dependent features across multiple variables) with 1D CNNs (to aggregate features across time).
-Feature extraction is performed per variable first, then combined across time.
-Grad-CAM was applied to intermediate layers to identify which variables and features contribute most to specific predictions, improving interpretability.

**Action Items for Next Week**
-Test using actual historical ramp data instead of the decision log and evaluate performance.
-Investigate the data copying pipeline to ensure there is no data leakage in testing.
-Conduct an independent experiment on feature importance, implementing interpretability methods to better understand model decisions.
-Begin drafting LaTeX notes, which will contribute to the IPAC proceedings.
-Run additional experiments to:
--Improve model performance
--Identify and apply more appropriate evaluation metrics


### **Meeting - 03/03/2026 - Erdong and Sai (In-person)**

**PROGRESS**

**Tested single-channel importance by:**
- Comparing plots between actual test data vs. simulations
- Simulating ramps using only one sensor + previous ramp inputs

**Missing-value imputation experiments:**
- Tried imputing with 0 and -100 (after normalization)
- Normalization used: z-score normalization, computed excluding NaN values

**Findings from imputation tests:**
- **0 imputation:** Even after removing all sensors except prev ramp (and imputing the removed sensors with 0), the simulated ramp looked similar to the original
- **-100 imputation:** Simulations looked similar when removing a single sensor, but not similar when removing multiple sensors

**Time-lag correlation analysis:**
- Plotted correlations between sensors
Found median lead/lag:
- Radiation leads Current by ~25sec
- Pressure leads Current by ~3sec
- Suggests time-lagged relationships between variables

**Discussion**
- Prior idea: imputing missing values with the previous value may behave similarly to 0 imputation, which could cause prev ramp to dominate training/testing
- -100 imputation may be too extreme and could distort the data
- Tested -15, saw similar behavior to -100 → suggests “extreme negative” values may have similar effects
**Potential logic/code issue:**
- In the 0-imputation case, if input windows become “flat/similar,” it’s unclear how the model produces different next-step predictions
- This suggests a need to verify whether inputs are truly identical and whether preprocessing/imputation is applied as expected
- More conclusions should wait until debugging confirms the pipeline is correct

**Actions**
- Debug the code to confirm missing values are being imputed correctly (and at the correct stage in preprocessing)
Read more about:
- Channel-importance testing techniques
- Best practices for missing-value imputation in time-series / multivariate sensor data
Investigate discrepancy:
- Understand why the output of conv1 showed different channel importance but the final model output is different.
- Check for correlations, pipeline mistakes, or implementation issues and be ready to defend the findings

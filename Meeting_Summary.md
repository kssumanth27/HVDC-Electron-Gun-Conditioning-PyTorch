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

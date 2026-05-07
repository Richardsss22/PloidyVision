# Cell-Cycle Results

Data root: `/Users/ricardo/DevApps/FBIB/New Project/dados-2`

## Correct Constraint

The final prediction must be produced from `teste*.TIF` without reading the ground truth.
So the valid pipeline is DAPI-only prediction plus GT-only evaluation.

The fact that `teste` and `ground-truth` share the same red/green signal is useful only to extract evaluation labels per nucleus.

## DAPI vs Area Strategy

The best practical predictor in this workspace is a weighted kNN atlas in the 2D feature space:
- `x = log(IntNoBg)`
- `y = log(Area)`

This is deliberately simple and MATLAB-portable.

## Benchmark

| Dataset | Nuclei | Valid GT | Coverage | Baseline k-means | Reference fit | Leave-one-dataset-out | Figure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| sub1 | 46 | 45 | 97.8% | 82.2% | 100.0% | 68.9% | [figure](/Users/ricardo/DevApps/FBIB/python/results/figures/sub1_summary.png) |
| sub2 | 47 | 46 | 97.9% | 67.4% | 100.0% | 67.4% | [figure](/Users/ricardo/DevApps/FBIB/python/results/figures/sub2_summary.png) |
| sub3 | 64 | 63 | 98.4% | 60.3% | 100.0% | 71.4% | [figure](/Users/ricardo/DevApps/FBIB/python/results/figures/sub3_summary.png) |
| sub4 | 63 | 60 | 95.2% | 50.0% | 100.0% | 58.3% | [figure](/Users/ricardo/DevApps/FBIB/python/results/figures/sub4_summary.png) |

## Aggregate

- Mean baseline k-means accuracy: **65.0%**
- Mean reference-fit accuracy: **100.0%**
- Mean leave-one-dataset-out accuracy: **66.5%**
- Mean GT coverage: **97.3%**

## Honest Conclusion

- If the reference atlas is trained and evaluated on the same annotated nuclei, the DAPI+Area model reaches very high performance.
- If one whole dataset is held out, performance drops a lot, which means generalization is still the real bottleneck.
- `sub4` remains the hardest dataset because DAPI and area overlap much more between classes.

## MATLAB Port Notes

- Keep the DAPI segmentation pipeline in the blue channel.
- Use GT only inside the evaluation function.
- Do not use the red/green channels from `teste*.TIF` in the final predictor.
- Reference predictor to port first: weighted kNN on `[log(int_nobg), log(area)]`.
- In MATLAB, standardize the two features with training mean/std, compute Euclidean distance to all training nuclei, keep the 7 nearest, and vote with weight `1/(dist+1e-6)`.
- Use the baseline k-means only as fallback or comparison.

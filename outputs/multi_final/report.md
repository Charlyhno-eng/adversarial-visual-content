# Local evaluation of visual signatures

Selected candidate: **smooth_08**. PNG SSIM: **0.98050**.

**All local detectors separated in every test: False**.
If this result is false, the benchmark objective is not achieved. A large hash distance or low feature similarity does not mean “a new image for Instagram/TikTok.”

## Protocol

- Working reference: [385, 512], source preserved separately.
- Budgets: [2, 4, 8] levels out of 255; 160 iterations per strategy.
- Minimum PNG quality: SSIM ≥ 0.98. This does not certify invisibility.
- Strategies: approximate DCT signatures on pixels or smooth fields, and signatures + ResNet-18/MobileNet representations.
- Optimization with rounding, JPEG 80/90 (approximate gradient), and resizing to 80%.
- Actual evaluation after PNG export, JPEG 90, JPEG 75, and resizing to 75%. The reference signature remains that of the initial clean image for every test.
- PDQ: separation if distance > 31, quality ≥ 50, including the closest match across the candidate's eight orientations.
- pHash/dHash: separation if distance > 10 bits out of 64, a local convention.
- Two deep features: separation if cosine < 0.8, a local convention; these are not probabilities.
- All criteria must pass simultaneously across the four transformations for local success. Selection prioritizes candidates meeting the quality constraint, then the minimum number of separated perceptual signatures across transformations, then the minimum and mean number of separated detectors, and finally SSIM.

Instagram's and TikTok's models and settings, content indexes, and recommendation signals are not accessible here. No upload was performed, and no bypass of these platforms was validated. Audio, temporal sequences, OCR, and metadata are not tested. This system processes still images, not videos.

See `measurements.csv` for all measurements and `report.json` for parameters, versions, and the source SHA-256.

## Sources

- [PDQ: recommended distances and quality](https://github.com/facebook/ThreatExchange/blob/main/pdq/README.md).
- [PDQ bindings used for actual evaluation](https://pypi.org/project/pdqhash/).
- [ImageHash: pHash and dHash](https://github.com/JohannesBuchner/imagehash).
- [Torchvision: pretrained models](https://docs.pytorch.org/vision/stable/models.html).

The DCT surrogate is an approximation used to guide optimization, not a certified differentiable reproduction of PDQ. PDQ measurements come exclusively from `pdqhash` on exported pixels. No protection against training is demonstrated.

# Raw_data_and_benchmarks

Raw measurement data, analysis scripts, and numerical outputs for the quantitative benchmarks.
This folder regenerates Fig. 2f,g and Fig. 3m–p of the main text and Figs. S2, S10, S18, S19 of the
Supplementary Information, together with every number quoted in Supplementary Sections S7, S13, S15,
and S16.

Requirements: Python 3 with `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-image`, `Pillow`,
`openpyxl`. All scripts run from any working directory; the two scripts that take an argument expect
the repository top-level directory.

## Fig2_two_point_ML_CRB — measured two-point data, maximum-likelihood fits, Cramér–Rao bounds

Raw data (`data/`):

- `optics_fields/1.png … 10.png` — the ten repeated full-field acquisitions of the microfabricated
  two-point samples under broadband incoherent illumination. Each field contains all twenty
  separations (0.05–0.24 mm in 0.01 mm steps), laid out row-major in 900 × 900-pixel tiles at
  916.2 pixels/mm: tile (row r, column c) holds the separation 0.05 + 0.01 × (5r + c) mm.
- `optics_labeled_crops/` — the per-separation labeled crops used to verify the tile layout
  (file name = separation in mm).
- `fig2_tilemap.json` — inter-acquisition registration shifts and the matched crop positions
  (produced by `optics_step1_register_tilemap.py`).
- `ultrasound_cscan/` — the thirty measured ultrasound C-scan maps, named
  `<separation mm>_<repeat>.xlsx`, 0.05 mm scan step (3.5 MHz transducer; see paper Methods).

Scripts, in order:

1. `optics_step1_register_tilemap.py` — registers the ten fields and template-matches the labeled
   crops to establish the tile layout (output already included as `data/fig2_tilemap.json`).
2. `optics_step2_ml_fits.py` — fits every tile of every field with the two-point forward model
   (planar background + two Gaussian spots, shared width, free orientation; ten parameters) and
   computes the Cramér–Rao bound for the separation from the fitted Jacobian and the measured
   residual noise → `out/fig2_optics_fits.csv`, `out/fig2_optics_tile_summary.csv`.
3. `optics_step3_scalar_calibration.py` — reproduces the single-mode virtual-photocurrent
   calibration of Fig. 2b from the same raw fields → `out/fig2_optics_scalar.csv`
   (correlation with the Fig. 2b values: r = 0.98).
4. `ultrasound_ml_fits.py` — the same two-point fit and CRB on the thirty C-scan maps →
   `out/fig2_ultrasound_fits.csv`, `out/fig2_ultrasound_summary.csv`.
5. `make_fig2_fg.py` — main-text Fig. 2f,g from the tables above → `../figures/fig2_fg.png`.
6. `make_figS18.py` — Supplementary Fig. S18 (six-panel fit/CRB summary) → `../figures/figS18.png`.
7. `make_figS17_envelope.py` — Supplementary Fig. S17: envelope-resolution (Rayleigh dip-contrast)
   analysis and estimator calibration on the same raw data (Supplementary Section S14); prints the
   dip-contrast and FWHM values quoted there → `../figures/figS17.png`.

Notes: the smallest ultrasound separation (0.085 mm) has two archived repeats and an unstable fit;
it is excluded from the quantitative panels, as stated in Supplementary Section S15. The optical
fit develops a floor and loses monotonicity below 0.10 mm (shaded band in Fig. 2f).

## Fig3_benchmark — image-quality benchmark and Wiener consistency

- `bench_fig3.py <repo_root> <out_dir>` — runs all ten readouts (direct imaging; physical mode
  sorting 20×20 and 35×35, noiseless; virtual mode sorting 20×20 with and without clipping;
  Wiener, Richardson–Lucy + TV, ISTA, FISTA, TV, each with an oracle hyperparameter sweep) on
  identical noisy realizations of the CU-letter object (Gaussian PSF σ = 0.5 mm, Poisson shot
  noise, N = 8 seeds at peak photon numbers 10¹²–10¹⁸ per pixel) →
  `out/fig3_benchmark_raw.xlsx`, `out/fig3_benchmark_summary.xlsx`.
  Example: `python bench_fig3.py ../.. out`
- `make_fig3_mp_figS19.py` — main-text Fig. 3m–p and Supplementary Fig. S19 from the benchmark
  tables → `../figures/fig3_mp.png`, `../figures/figS19.png`.
- `make_figS9.py` — Supplementary Fig. S9 (DI, PMS 20×20, PMS 35×35, VMS 20×20 and
  Richardson–Lucy at peak 10¹⁸) from the same benchmark tables → `../figures/figS9.png`.
- `wiener_check.py <repo_root>` — the Wiener consistency check of Supplementary Section S7: the
  oracle-tuned scalar-λ Wiener filter reaches 12.4 dB PSNR on this object at every photon budget
  and on noise-free data (prior-mismatch limited); the bare inverse filter collapses to 5.8 dB;
  virtual mode sorting reaches 13.6 dB at peak 10¹⁸.
- `make_figS10.py <repo_root>` — Supplementary Fig. S10 (Wiener vs virtual mode sorting at low and
  high noise) → `../figures/figS10.png`.
- `make_figS2.py` — Supplementary Fig. S2 (classical Fisher information of direct imaging vs
  binary SPADE, normalized to the quantum Fisher information) → `../figures/figS2.png`.

## figures

The generated figure panels as used in the manuscript: `fig2_fg.png`, `fig3_mp.png`, `figS2.png`,
`figS9.png`, `figS10.png`, `figS17.png`, `figS18.png`, `figS19.png`.

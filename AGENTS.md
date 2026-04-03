# Project Rules for Codex

## Task Definition
- DiffTraffic studies `short-term spatiotemporal traffic forecasting`.
- The primary task is deterministic point forecasting.
- The method innovation track is residual refinement, with probabilistic or diffusion-style refinement allowed only after the deterministic baseline is sound.
- Do not pivot the main project to long-term generic forecasting, classification, anomaly detection, or multi-task foundation modeling.

## Scope
- Keep all new DiffTraffic work under `src/basicts/models/DiffTraffic/`.
- Baseline reproduction work that is necessary for the corrected traffic mainline may be added under `src/basicts/models/<BaselineName>/`.
- Reproduce `STAEformer` inside BasicTS as the current target deterministic backbone for the traffic mainline.
- Use the existing `STID` implementation from `src/basicts/models/STID/` only as a sanity baseline without editing its source files.
- Treat `TimeMixer-on-traffic` as an archived negative control, not the mainline.
- Keep the current BasicTS forecasting training flow unless compatibility forces a thin wrapper.
- Do not add adaptive noise or retrieval guidance in v0 or v1.

## Architecture Rules
- `BackboneAdapter` must only wrap a deterministic backbone and expose `base_prediction`.
- The current target deterministic backbone for the mainline is `STAEformer`; if an adapter is added for the mainline, it should wrap `STAEformer` explicitly.
- `STID` may be wrapped only for sanity-check experiments or ablations.
- `ConditionEncoderStack` or equivalent modules may compute condition features.
- `ConditionFusion` must only organize already-computed conditioning inputs.
- `ResidualRefiner` or `ResidualDiffusionDecoder` must only produce the residual prediction `R_hat`.
- `DiffTrafficForForecasting` must combine `Y_base` and `R_hat` as `prediction = base_prediction + residual_prediction`.
- Keep `base_prediction`, `residual_prediction`, and `aux_info` in the model return dictionary.

## Experiment Rules
- Do not draw method conclusions from a single dataset.
- The minimum benchmark pair for the mainline is `METR-LA` and `PEMS-BAY`.
- Evaluate traffic forecasting in raw scale with `rescale=True` when comparing to papers.
- Prioritize `MAE` and `RMSE` as primary report metrics; treat `MAPE` as secondary and `WAPE` as diagnostic only.
- Before residual-model work is considered merge-ready, reproduce the pure deterministic `STAEformer` baseline in BasicTS on the benchmark pair.
- Maintain `STID` as a low-cost sanity baseline to verify training and evaluation plumbing.
- Do not adopt a new mainline backbone unless it passes:
  - literature-fit validation for short-term traffic forecasting
  - benchmark-fit validation on the traffic benchmark pair
  - negative-control reasoning showing why the chosen inductive bias is necessary
- `T-Graphormer` is a valid stage-two comparison target, but it is too heavy to be the first reproduced mainline backbone.

## Editing Rules
- Do not modify any `STID` source files.
- If `STAEformer` reproduction starts, keep it isolated in its own model directory and do not couple its code to `DiffTraffic`.
- Do not resume feature work on the `TimeMixer` traffic branch unless explicitly asked to maintain the archived negative control.
- Prefer small, isolated files and keep interfaces explicit.
- Add shape comments for every public `forward` method.
- Keep the early DiffTraffic implementation lightweight and interface-first.

## Validation Rules
- After adding files, validate with import checks first.
- Then validate model construction and one minimal forward pass.
- Then validate one real traffic single-step train step.
- Do not run full residual-model training until the pure `STAEformer` baseline is reproducible on the benchmark pair.
- Every major design change must be accompanied by at least one explicit falsification check against the task definition or backbone assumption.

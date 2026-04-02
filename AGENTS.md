# Project Rules for Codex

## Scope
- Keep all new DiffTraffic work under `src/basicts/models/DiffTraffic/`.
- Reuse the existing TimeMixer implementation from `src/basicts/models/TimeMixer/` without editing its source files.
- Keep the current BasicTS forecasting training flow unless compatibility forces a thin wrapper.
- Do not add adaptive noise or retrieval guidance in v0.

## Architecture Rules
- `TimeMixerAdapter` must only wrap the existing TimeMixer backbone and expose `base_prediction`.
- `ResidualDiffusionDecoder` must only produce the residual prediction `R_hat`.
- `ConditionFusion` must only organize conditioning inputs.
- `DiffTrafficV0ForForecasting` must combine `Y_base` and `R_hat` as `prediction = base_prediction + residual_prediction`.
- Keep `base_prediction`, `residual_prediction`, and `aux_info` in the model return dictionary.

## Editing Rules
- Do not modify any `TimeMixer` source files.
- Prefer small, isolated files and keep interfaces explicit.
- Add shape comments for every public `forward` method.
- Keep v0 implementation lightweight and interface-first.

## Validation Rules
- After adding files, validate with import checks first.
- Then validate model construction and one minimal forward pass.
- Do not run full training for v0 shell work.

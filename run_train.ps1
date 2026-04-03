Set-Location -Path $PSScriptRoot

New-Item -ItemType Directory -Path ".\runs\difftraffic" -Force | Out-Null

$env:PYTHONPATH = "src"

.\conda_envs\basicts\python.exe -u -c "from basicts import BasicTSLauncher; from basicts.configs import BasicTSForecastingConfig; from basicts.models.DiffTraffic import DiffTrafficV0ForForecasting, DiffTrafficV0Config, difftraffic_loss; cfg=BasicTSForecastingConfig(model=DiffTrafficV0ForForecasting, model_config=DiffTrafficV0Config(input_len=12, output_len=12, num_features=207, hidden_size=32, num_layers=1, down_sampling_layers=1, down_sampling_window=2, moving_avg=3, use_timestamps=True, timestamp_sizes=[288,7], residual_hidden_size=64, residual_dropout=0.0), dataset_name='METR-LA', input_len=12, output_len=12, use_timestamps=True, gpus=None, num_epochs=100, loss=difftraffic_loss); BasicTSLauncher.launch_training(cfg)" *> ".\runs\difftraffic\train_log.txt"

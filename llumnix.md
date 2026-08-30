python3 -m vidur.main \
  --global_scheduler_config_type llumnix \
  --llumnix_global_scheduler_config_enable_migration \
  --cluster_config_num_replicas 2 \
  --synthetic_request_generator_config_num_requests 20 \
  --time_limit 100 \
  --metrics_config_enable_chrome_trace \
  --log_level debug

  
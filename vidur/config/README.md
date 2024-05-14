# Understanding the parameters taken by the simulator

The [default.yml](vidur/config/default.yml) is the comprehensive list of all parameters taken by the simulator. While invoking the simulator, any of these parameters can be overrided. Running only `python -m vidur.main` means that all the parameters are taken from the `default.yml` file and no overrides.
The parameters descriptions are given below:

1. `seed`: Random seed which is set in multiple random generators notably the request length and inter-request time generators. This is useful for reproducibility.
2. `log_level`: Logging level for the simulator. Not comprehensively supported currently.
3. `output_dir`: The directory which each invocation of the simulator creates its directory under. Eg: `./simulator_output/2023-11-20_11-31-40-523377`.
All the output files corresponding to the invocation are stored under this directory eg. the chrome trace, cdf plots etc.
4. `cache_dir`: The simulator has tiny models inside it to predict time taken by various model operations eg. `mlp_up_proj`. These model weights are cached in this directory.
5. `write_json_trace`: Whether to write the requests sent to the simulator in a json file.
6. `write_chrome_trace`: Whether to write the chrome trace. This is useful for debugging. Use `chrome://tracing` or `edge://tracing` to view the trace.
7. `write_metrics`: This is a blanket flag to enable/disable writing of all metrics. Should be set to `true` as metrics are the only thing that the simulator gives, no LLM is actually running inside it.
8. `cluster`:
    1. `num_replicas`: Number of replicas in the clusters. Replicas and independent and identical.
    Suppose you have a DGX box with 8 GPUS and you want to serve `meta-llama/Llama-2-70b-hf`.
    One deployment strategy is to run 2 replicas each with 4 GPUs running the model in tensor parallel degree 4.
    Another deployment strategy is to run 1 replica with all the 8 GPUs running the model in tensor parallel degree 8.
9. `replica`: Configuration of each replica.
    1. `block_size`: This is a concept from vLLM. Each request has a number of tokens each of whose KV value needs to be cached. The cache is divided into blocks of size `block_size`. The number of blocks each request needs is `num_blocks = ceil(num_tokens / block_size)`.
    2. `memory_margin_fraction`: From vLLM. Fraction of memory that is left unused typically for `nccl`, `cuBLAS` libraries. This is not a strict constraint. Actual deployment does go over this limit.
    3. `num_pipeline_stages`: Pipeline parallel degree. This number must divide the number of layers in the model.
    4. `num_tensor_parallel_workers`: Tensor parallel degree.
    5. Model specs: Refer huggingface `config.json` for the model eg. <https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/config.json>
        1. `model_name`: Typically huggingface id of the model. Eg: `meta-llama/Llama-2-70b-hf`. Custom model architectures can be used but please take the resultd with a large grain of salt.
        2. `num_layers`
        3. `num_q_heads`
        4. `num_kv_heads`
        5. `embedding_dim`
        6. `mlp_hidden_dim`
        7. `use_gated_mlp`
        8. `vocab_size`
    6. GPU specs:
        1. `fp16_tflops`: TFLOPS of the GPU in FP16. This is used to predict the execution time of the model.
        2. `total_memory_gb`: Total memory of the GPU in GB. This is used in memory calculations of the model weights, KV cache etc.
        3. For `a100`: `fp16_tflops: 312`, `total_memory_gb: 80`
10. `request_generator`: The simulator contains a comprehensive request generator. See [here](vidur/request_generator)
    1. `provider`: The request generator to use. Currently supported are `synthetic`, `trace`. `synthetic` generates requests from a synthetic distribution. `trace` generates requests from a real-world trace.
    2. `max_tokens`: Maximum number of tokens in a request. Requests generated from the trace are capped / clipped at this number. `P:D ratio` is preserved in case of clipping.
11. `synthetic_request_generator`: This section is used to further define the synthetic request generator. Only required if `request_generator_provider` is set to `synthetic`.
    1. `length_provider`: The distribution of the request length. Currently supported are `uniform`, `trace` and `zipf`.
    2. `interval_provider`: The distribution of the inter-request time. Currently supported are `static`, `trace`, `poisson` and `gamma`.
    3. `min_tokens`: Minimum number of tokens in a request when `uniform`, `zipf` is used as the `length_provider`. TODO: Verify for `trace` as well.
    4. `prefill_to_decode_ratio`: Ratio of prefill tokens to decode tokens in a request. This is used in `uniform` length provider. TODO: Verify for `zipf` as well.
    5. `num_requests`: Number of requests to generate / select from the trace.
12. `trace_request_generator`: This section is used to to further define the trace request generator. Only required if `request_generator_provider` is set to `trace`.
    1. `trace_file`: Path to the trace file.
    2. `date`: Date of the trace to use.
    3. `prefill_scale_factor`: Scale factor to apply to the prefill tokens in the trace. Recommend to leave this value at 1.
    4. `decode_scale_factor`: Scale factor to apply to the decode tokens in the trace. Recommend to leave this value at 1.
    5. `time_scale_factor`: Scale factor to apply to the window time in the trace. This can be used to speed up / slow down the trace. Example, to compress a 24h trace to 1h. Scale factors drastically change the worload. Scaled traces cannot be typically directly compared to the original trace.
13. `trace_request_length_generator`: Only required if `request_generator_provider` is set to `synthetic` and `synthetic_request_length_provider` is set to `trace`.
    1. `trace_file`: Path to the trace file. This trace file is a csv like [cnn_dailymail_stats_llama2_tokenizer.csv](data/processed_traces/cnn_dailymail_stats_llama2_tokenizer.csv)
    2. `prefill_scale_factor`: See `trace_request_generator` section above.
    3. `decode_scale_factor`: See `trace_request_generator` section above.
14. `trace_request_interval_generator`: Only required if `request_generator_provider` is set to `synthetic` and `synthetic_request_interval_provider` is set to `trace`.
    1. `trace_file`: Path to the trace file.
    2. `start_time`: Start time of the trace to use.
    3. `end_time`: End time of the trace to use.
    4. `time_scale_factor`: See `trace_request_generator` section above.
15. `poisson_request_interval_generator`: Only required if `request_generator_provider` is set to `synthetic` and `synthetic_request_length_provider` is set to `poisson`.
    1. `qps`: Requests per second to hit the system with.
16. `gamma_request_interval_generator`: Only required if `request_generator_provider` is set to `synthetic` and `synthetic_request_length_provider` is set to `gamma`.
    1. `cv`: Coefficient of variation of the gamma distribution.
    2. `qps`: Requests per second to hit the system with.
17. `zipf_request_length_generator`: Only required if `request_generator_provider` is set to `synthetic` and `synthetic_request_length_provider` is set to `zipf`.
    1. `theta`: Shape parameter of the zipf distribution.
    2. `scramble`: Whether to scramble the zipf distribution. This is useful to avoid the zipf distribution to be skewed towards the start of the vocabulary.
18. `execution_time_predictor`: Type of the tiny models inside the simulator to predict the execution time of the model.
    1. `provider`: `sklearn`, `random_forrest` or `linear_regression`.
19. `sklearn_execution_time_predictor`:
    1. `compute_input_file`: `./data/profiling/a100/mlp.csv`
    2. `attention_input_file`: `./data/profiling/a100/mixed_attention.csv`
    3. `all_reduce_input_file`: `./data/profiling/a100/all_reduce.csv`
    4. `send_recv_input_file`: .`/data/profiling/a100/p2p_intra_node.csv`
    5. `cpu_overhead_input_file`: `./data/profiling/a100/cpu_overheads.csv`
    6. `k_fold_cv_splits`: `10`
    7. `no_cache`: `false`
    8. `kv_cache_prediction_granularity`: `8`
    9. `prediction_max_prefill_chunk_size:`4096`
    10. `prediction_max_batch_size`: `100`
    11. `prediction_max_tokens_per_request`: `4096`
    12. `attention_decode_overhead_percentage`: `0.0`
    13. `nccl_cpu_launch_overhead_ms`: `0.020`
20. `random_forrest_execution_time_predictor`: TODO. Recommend to use the `sklearn_execution_time_predictor` instead.
21. `linear_regression_execution_time_predictor`: TODO. Recommend to use the `sklearn_execution_time_predictor` instead.
22. `simulator`:
    1. `time_limit`: Time limit for the simulator to run. This is useful to run the simulator for a fixed amount of time. The simulator will stop after this time limit is reached. Default is no limit. TODO: Verify the functionality of this parameter.
23. `global_scheduler`: This is the scheduler which determines which replica to send the request to.
    1. `provider`: `round_robin`, `random`, `lor`. See [here](vidur/schedulers/global_schedulers) for more details.
24. `replica_scheduler`: This is the scheduler which determines how to schedule the requests on a replica.
    1. `provider`: `orca`, `sarathi`, and `vllm`. See [here](vidur/schedulers/replica_schedulers) for more details.
    2. `batch_size_cap`: Maximum permissible batch size. Set carefully for `orca`. Have a high limit for other schedulers. They will auto-adjust.
    3. `num_blocks`: TODO. Ignore this parameter for now.
25. `orca_scheduler`: Only required if `replica_scheduler_provider` is set to `orca`.
    1. `use_single_prefill_per_batch`: Whether to use a single prefill per batch. This is a non-standard param that if true `orca` scheduler quite competitive.
26. `sarathi_scheduler`: <https://arxiv.org/abs/2308.16369>. Only required if `replica_scheduler_provider` is set to `sarathi`.
    1. `chunk_size`: The maximum number of tokens (prefill / decode) to process in a batch. Prefills are done progressively if the number of prefills tokens in a request is greater than this number.
    2. `enable_rolling_prefills`: Multiple prefills are done in a batch provided sum of the prefills is less than `chunk_size`.
    3. `prefill_fitting_tolerance`: Ignore this parameter. Leave it at 0.0.
27. `vllm_scheduler`: <https://github.com/vllm-project/vllm>. Only required if `replica_scheduler_provider` is set to `vllm`.
    1. `watermark_blocks_fraction`: If this param is 0.01, then we consider the cache is full when 99% of the blocks are full. Prevents unnecessary swaps.
    2. `max_tokens_in_batch`: Maximum number of tokens in a batch. This is an additional limit on top of `batch_size_cap`.
    3. `max_batch_size_amplification_factor`: Ignore this parameter, leave it at `1`.
29. `metrics_store`: Configuration of the metrics store. The metrics store is a cental store which stores the metrics of the simulator. At simulation end, it dumps the metrics to various files typically `csv`, `png` and `json`. The metrics store is also responsible for uploading the metrics to `wandb`.
    1. `wandb_project`: Wandb project to upload to eg. `llm-simulator`
    2. `wandb_group`
    3. `wandb_run_name`: Leave empty string to auto-generate the run name. Recommend to have a run name to identify the run.
    4. `subsamples`: Number of subsamples to take from the metrics. This is useful to limit the number of datapoints of a metric.
    5. `save_table_to_wandb`: Whether to upload the csvs corresponding to the plots uploaded to wandb. Set it to true.
    6. `min_batch_idx`: Ignore this parameter.
    7. `max_batch_idx`: Ignore this parameter.

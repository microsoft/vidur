
#!/usr/bin/bash

set -ex

if [ $# -eq 0 ]
then
	echo "Please provide group name as input"
	exit
fi

if [ -z "$1" ]
then
	echo "Input must be string"
	exit
fi

GROUP_NAME=$1
echo "Group name is : " $GROUP_NAME

SIMULATOR="python -m simulator.main"

mkdir -p logs

COMMON_ARGS=\
"--metrics_store_wandb_group $GROUP_NAME \
--replica_num_layers 80 \
--replica_num_q_heads 232 \
--replica_num_kv_heads 8 \
--replica_embedding_dim 14848 \
--replica_mlp_hidden_dim 59392 \
--no-replica_use_gated_mlp \
--replica_vocab_size 65024 \
--replica_fp16_tflops 312 \
--sklearn_execution_time_predictor_send_recv_input_file ./data/profiling/a100/p2p_intra_node.csv \
--replica_total_memory_gb 80
--cluster_num_replicas 8 \
--replica_num_tensor_parallel_workers 8 \
--replica_num_pipeline_stages 1"

# iterate over different batch sizes
# by default this config suppport batch size of ~300,
# so the batch size is purely determined by the cap we set here
MAX_BATCH_SIZES=(8 16 32 64 128)

for MAX_BATCH_SIZE in ${MAX_BATCH_SIZES[@]}
do
	$SIMULATOR \
	$COMMON_ARGS \
	--replica_scheduler_provider "orca" \
	--replica_scheduler_batch_size_cap $MAX_BATCH_SIZE \
	--metrics_store_wandb_run_name "orca tp:8 pp:1 bsz:$MAX_BATCH_SIZE" &>  logs/orca_tp8_pp1_bsz_$MAX_BATCH_SIZE.log &

	$SIMULATOR \
	$COMMON_ARGS \
	--replica_scheduler_provider "vllm" \
	--replica_scheduler_batch_size_cap $MAX_BATCH_SIZE \
	--metrics_store_wandb_run_name "vllm tp:8 pp:1 bsz:$MAX_BATCH_SIZE" &>  logs/vllm_tp8_pp1_bsz_$MAX_BATCH_SIZE.log &

	$SIMULATOR \
	$COMMON_ARGS \
	--replica_scheduler_provider "dsarathi" \
	--dsarathi_scheduler_chunk_size 256 \
	--replica_scheduler_batch_size_cap $MAX_BATCH_SIZE \
	--metrics_store_wandb_run_name "dsarathi tp:8 pp:1 bsz:$MAX_BATCH_SIZE cs:256" &>  logs/dsarathi_tp8_pp1_bsz$MAX_BATCH_SIZE_cs256.log &
	
	$SIMULATOR \
	$COMMON_ARGS \
	--replica_scheduler_provider "dsarathi" \
	--dsarathi_scheduler_chunk_size 512 \
	--replica_scheduler_batch_size_cap $MAX_BATCH_SIZE \
	--metrics_store_wandb_run_name "dsarathi tp:8 pp:1 bsz:$MAX_BATCH_SIZE cs:512" &>  logs/dsarathi_tp8_pp1_bsz$MAX_BATCH_SIZE_cs512.log &
	
	$SIMULATOR \
	$COMMON_ARGS \
	--replica_scheduler_provider "dsarathi" \
	--dsarathi_scheduler_chunk_size 1024 \
	--replica_scheduler_batch_size_cap $MAX_BATCH_SIZE \
	--metrics_store_wandb_run_name "dsarathi tp:8 pp:1 bsz:$MAX_BATCH_SIZE cs:1024" &>  logs/dsarathi_tp8_pp1_bsz$MAX_BATCH_SIZE_cs1024.log &

	wait
done


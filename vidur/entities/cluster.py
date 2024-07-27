import json

from vidur.config import ClusterConfig
from vidur.entities.base_entity import BaseEntity
from vidur.entities.replica import Replica
from vidur.logger import init_logger

logger = init_logger(__name__)


class Cluster(BaseEntity):
    def __init__(self, cluster_config: ClusterConfig) -> None:
        self._id = Cluster.generate_id()
        self._config = cluster_config

        # Init replica object handles
        self._replicas = {}

        for _ in range(self._config.num_replicas):
            replica = Replica(self._config.replica_config)
            self._replicas[replica.id] = replica

        if self._config.write_json_trace:
            self._write_cluster_info_to_file()

    @property
    def replicas(self):
        return self._replicas

    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "num_replicas": len(self._replicas),
        }

    def _write_cluster_info_to_file(self) -> None:
        replica_dicts = [replica.to_dict() for replica in self._replicas.values()]
        cluster_info = {"replicas": replica_dicts}

        cluster_file = f"{self._config.output_dir}/cluster.json"
        with open(cluster_file, "w") as f:
            json.dump(cluster_info, f)

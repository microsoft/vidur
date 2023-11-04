import json
import logging

from simulator.config import Config
from simulator.entities.base_entity import BaseEntity
from simulator.entities.replica import Replica

logger = logging.getLogger(__name__)


class Cluster(BaseEntity):
    def __init__(self, config: Config):
        self._id = Cluster.generate_id()
        self._config = config

        # Init replica object handles
        self._replicas = {}

        for _ in range(self._config.cluster_num_replicas):
            replica = Replica(config)
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
        json.dump(cluster_info, open(cluster_file, "w"))

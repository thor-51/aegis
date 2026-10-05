"""
aegis/k8s/actuator.py — executes AEGIS actions (NOOP, SCALE_UP, SCALE_DOWN, RESTART, MIGRATE)
against a live Kubernetes cluster via the official Kubernetes Python client.
"""

from __future__ import annotations

import datetime
import logging
from typing import Sequence

try:
    from kubernetes import client, config
    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False

from aegis.env.microservice_env import ActionType

logger = logging.getLogger(__name__)


class K8sActuator:
    """
    Translates discrete AEGIS actions into Kubernetes API mutations.
    """

    def __init__(
        self,
        service_names: Sequence[str],
        namespace: str = "aegis-demo",
        min_replicas: int = 1,
        max_replicas: int = 10,
        in_cluster: bool = False,
    ):
        self.service_names = list(service_names)
        self.namespace = namespace
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas

        if K8S_AVAILABLE:
            try:
                if in_cluster:
                    config.load_incluster_config()
                else:
                    config.load_kube_config()
                self.apps_v1 = client.AppsV1Api()
                self.core_v1 = client.CoreV1Api()
            except Exception as e:
                logger.warning("Could not initialize Kubernetes client: %s", e)
                self.apps_v1 = None
                self.core_v1 = None
        else:
            self.apps_v1 = None
            self.core_v1 = None

    def execute_action(self, service_id: int, action_type: ActionType | int) -> dict:
        """
        Executes the requested action on the given service.
        Returns a dict summarizing the action outcome.
        """
        if isinstance(action_type, int):
            action_type = ActionType(action_type)

        if not (0 <= service_id < len(self.service_names)):
            raise ValueError(f"Invalid service_id {service_id}, expected 0..{len(self.service_names)-1}")

        service_name = self.service_names[service_id]

        if action_type == ActionType.NOOP:
            return {"action": "NOOP", "service": service_name, "status": "ok"}

        if self.apps_v1 is None:
            # Running in dry-run or mock mode
            return {"action": action_type.name, "service": service_name, "status": "mock_executed"}

        try:
            if action_type == ActionType.SCALE_UP:
                return self._scale(service_name, delta=+1)
            elif action_type == ActionType.SCALE_DOWN:
                return self._scale(service_name, delta=-1)
            elif action_type == ActionType.RESTART:
                return self._restart(service_name)
            elif action_type == ActionType.MIGRATE:
                return self._migrate(service_name)
            else:
                return {"action": str(action_type), "service": service_name, "status": "unknown"}
        except Exception as e:
            logger.error("Failed to execute %s on %s: %s", action_type.name, service_name, e)
            return {"action": action_type.name, "service": service_name, "status": "error", "error": str(e)}

    def _scale(self, service_name: str, delta: int) -> dict:
        dep = self.apps_v1.read_namespaced_deployment(name=service_name, namespace=self.namespace)
        current_replicas = dep.spec.replicas or 1
        new_replicas = max(self.min_replicas, min(self.max_replicas, current_replicas + delta))

        act_name = "SCALE_UP" if delta > 0 else "SCALE_DOWN"
        if new_replicas == current_replicas:
            return {
                "action": act_name,
                "service": service_name,
                "status": "unchanged",
                "replicas": current_replicas,
            }

        body = {"spec": {"replicas": new_replicas}}
        self.apps_v1.patch_namespaced_deployment(name=service_name, namespace=self.namespace, body=body)
        return {
            "action": act_name,
            "service": service_name,
            "status": "ok",
            "old_replicas": current_replicas,
            "new_replicas": new_replicas,
        }

    def _restart(self, service_name: str) -> dict:
        """
        Triggers a rolling restart of the deployment by updating the kubectl.kubernetes.io/restartedAt annotation.
        """
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        body = {
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "kubectl.kubernetes.io/restartedAt": now
                        }
                    }
                }
            }
        }
        self.apps_v1.patch_namespaced_deployment(name=service_name, namespace=self.namespace, body=body)
        return {"action": "RESTART", "service": service_name, "status": "ok", "restarted_at": now}

    def _migrate(self, service_name: str) -> dict:
        """
        Simulates migration across worker nodes by deleting the oldest pod of the deployment,
        triggering a reschedule on available nodes.
        """
        pods = self.core_v1.list_namespaced_pod(
            namespace=self.namespace,
            label_selector=f"app={service_name}",
        ).items

        if not pods:
            return {"action": "MIGRATE", "service": service_name, "status": "no_pods_found"}

        # Target the oldest running pod
        oldest_pod = min(pods, key=lambda p: p.metadata.creation_timestamp or datetime.datetime.min)
        pod_name = oldest_pod.metadata.name
        self.core_v1.delete_namespaced_pod(name=pod_name, namespace=self.namespace)
        return {
            "action": "MIGRATE",
            "service": service_name,
            "status": "ok",
            "deleted_pod": pod_name,
        }

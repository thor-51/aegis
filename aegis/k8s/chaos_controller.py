"""
aegis/k8s/chaos_controller.py — injects faults (StressChaos, NetworkChaos, PodChaos)
via Chaos Mesh CRDs in Kubernetes.
"""

from __future__ import annotations

import logging
from typing import Sequence

try:
    from kubernetes import client
    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False

logger = logging.getLogger(__name__)


class ChaosController:
    """
    Manages Chaos Mesh Custom Resources to inject faults into services in the aegis-demo namespace.
    """

    GROUP = "chaos-mesh.org"
    VERSION = "v1alpha1"

    def __init__(self, service_names: Sequence[str], namespace: str = "aegis-demo"):
        self.service_names = list(service_names)
        self.namespace = namespace
        self.active_chaos = []

        if K8S_AVAILABLE:
            try:
                self.custom_api = client.CustomObjectsApi()
            except Exception:
                self.custom_api = None
        else:
            self.custom_api = None

    def inject_stress_cpu(self, service_name: str, duration: str = "30s") -> bool:
        """Injects CPU stress chaos."""
        chaos_name = f"cpu-burn-{service_name}"
        manifest = {
            "apiVersion": f"{self.GROUP}/{self.VERSION}",
            "kind": "StressChaos",
            "metadata": {"name": chaos_name, "namespace": self.namespace},
            "spec": {
                "mode": "all",
                "selector": {"namespaces": [self.namespace], "labelSelectors": {"app": service_name}},
                "stressors": {"cpu": {"workers": 2, "load": 90}},
                "duration": duration,
            },
        }
        return self._create_resource("stresschaos", chaos_name, manifest)

    def inject_network_delay(self, service_name: str, latency: str = "200ms", duration: str = "30s") -> bool:
        """Injects network latency chaos."""
        chaos_name = f"net-delay-{service_name}"
        manifest = {
            "apiVersion": f"{self.GROUP}/{self.VERSION}",
            "kind": "NetworkChaos",
            "metadata": {"name": chaos_name, "namespace": self.namespace},
            "spec": {
                "action": "delay",
                "mode": "all",
                "selector": {"namespaces": [self.namespace], "labelSelectors": {"app": service_name}},
                "delay": {"latency": latency},
                "duration": duration,
            },
        }
        return self._create_resource("networkchaos", chaos_name, manifest)

    def clear_all(self):
        """Cleans up all active chaos resources."""
        for plural, name in self.active_chaos:
            self._delete_resource(plural, name)
        self.active_chaos.clear()

    def _create_resource(self, plural: str, name: str, manifest: dict) -> bool:
        if not self.custom_api:
            logger.info("Mock inject %s: %s", plural, name)
            self.active_chaos.append((plural, name))
            return True

        try:
            self.custom_api.create_namespaced_custom_object(
                group=self.GROUP,
                version=self.VERSION,
                namespace=self.namespace,
                plural=plural,
                body=manifest,
            )
            self.active_chaos.append((plural, name))
            return True
        except Exception as e:
            logger.error("Failed to create %s %s: %s", plural, name, e)
            return False

    def _delete_resource(self, plural: str, name: str):
        if not self.custom_api:
            return
        try:
            self.custom_api.delete_namespaced_custom_object(
                group=self.GROUP,
                version=self.VERSION,
                namespace=self.namespace,
                plural=plural,
                name=name,
            )
        except Exception:
            pass

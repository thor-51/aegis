"""
aegis/k8s/observer.py — collects real-time metrics (CPU, memory, latency, error rate, replicas)
from Prometheus and Kubernetes API, normalizing them into the AEGIS observation space.
"""

from __future__ import annotations

import logging
from typing import Sequence
import numpy as np

try:
    from prometheus_api_client import PrometheusConnect
    PROM_AVAILABLE = True
except ImportError:
    PROM_AVAILABLE = False

try:
    from kubernetes import client, config
    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False

logger = logging.getLogger(__name__)


class K8sObserver:
    """
    Scrapes metrics for N microservices and computes a normalized 5-feature vector per service:
    [cpu_util, mem_util, latency_norm, error_rate, replica_count_norm].
    """

    def __init__(
        self,
        service_names: Sequence[str],
        prometheus_url: str = "http://localhost:9090",
        namespace: str = "aegis-demo",
        max_replicas: int = 10,
        max_latency_ms: float = 500.0,
    ):
        self.service_names = list(service_names)
        self.prometheus_url = prometheus_url
        self.namespace = namespace
        self.max_replicas = max_replicas
        self.max_latency_ms = max_latency_ms

        self.prom = None
        if PROM_AVAILABLE:
            try:
                import urllib.request
                # Fast 0.5s ping before enabling Prometheus client
                urllib.request.urlopen(f"{prometheus_url}/-/healthy", timeout=0.5)
                self.prom = PrometheusConnect(url=prometheus_url, disable_ssl=True)
                logger.info("Connected to Prometheus at %s", prometheus_url)
            except Exception as e:
                logger.info("Prometheus endpoint %s not reachable; using K8s metrics/fallback (%s)", prometheus_url, e)
                self.prom = None

        self.apps_v1 = None
        if K8S_AVAILABLE:
            try:
                self.apps_v1 = client.AppsV1Api()
            except Exception:
                pass

    def get_service_observation(self, service_name: str) -> np.ndarray:
        """
        Returns a normalized 5-element float array for the given service:
        [cpu_util, mem_util, latency_norm, error_rate, replica_norm].
        """
        cpu_util = self._query_cpu(service_name)
        mem_util = self._query_memory(service_name)
        latency_norm = self._query_latency(service_name)
        error_rate = self._query_error_rate(service_name)
        replica_norm = self._query_replicas(service_name)

        return np.array([cpu_util, mem_util, latency_norm, error_rate, replica_norm], dtype=np.float32)

    def get_full_observation(self, time_since_last_fault: float = 1.0) -> np.ndarray:
        """
        Returns the concatenated observation across all services + normalized time_since_last_fault.
        Shape: (len(service_names) * 5 + 1,)
        """
        obs_parts = []
        for name in self.service_names:
            obs_parts.append(self.get_service_observation(name))

        service_obs = np.concatenate(obs_parts)
        full_obs = np.append(service_obs, np.float32(np.clip(time_since_last_fault, 0.0, 1.0)))
        return full_obs.astype(np.float32)

    def _query_prom(self, query: str) -> float | None:
        if not self.prom:
            return None
        try:
            # Quick check if prometheus is reachable
            if hasattr(self, "_prom_reachable") and not self._prom_reachable:
                return None
            result = self.prom.custom_query(query=query)
            self._prom_reachable = True
            if result and len(result) > 0:
                val = result[0].get("value", [None, None])[1]
                return float(val) if val is not None else None
        except Exception as e:
            self._prom_reachable = False
            logger.debug("Prometheus query failed '%s': %s", query, e)
        return None

    def _query_cpu(self, service_name: str) -> float:
        query = f'sum(rate(container_cpu_usage_seconds_total{{namespace="{self.namespace}", pod=~"{service_name}-.*"}}[1m]))'
        val = self._query_prom(query)
        if val is None:
            return 0.3  # fallback baseline
        return float(np.clip(val, 0.0, 1.0))

    def _query_memory(self, service_name: str) -> float:
        query = f'sum(container_memory_usage_bytes{{namespace="{self.namespace}", pod=~"{service_name}-.*"}})/sum(kube_pod_container_resource_limits{{namespace="{self.namespace}", pod=~"{service_name}-.*", resource="memory"}})'
        val = self._query_prom(query)
        if val is None:
            return 0.3  # fallback baseline
        return float(np.clip(val, 0.0, 1.0))

    def _query_latency(self, service_name: str) -> float:
        query = f'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{{namespace="{self.namespace}", app="{service_name}"}}[1m])) by (le)) * 1000'
        val = self._query_prom(query)
        if val is None:
            return 0.1  # fallback ~50ms
        return float(np.clip(val / self.max_latency_ms, 0.0, 1.0))

    def _query_error_rate(self, service_name: str) -> float:
        query = f'sum(rate(http_requests_total{{namespace="{self.namespace}", app="{service_name}", status=~"5.."}}[1m])) / sum(rate(http_requests_total{{namespace="{self.namespace}", app="{service_name}"}}[1m]))'
        val = self._query_prom(query)
        if val is None or np.isnan(val):
            return 0.0
        return float(np.clip(val, 0.0, 1.0))

    def _query_replicas(self, service_name: str) -> float:
        if self.apps_v1:
            try:
                dep = self.apps_v1.read_namespaced_deployment(name=service_name, namespace=self.namespace)
                replicas = dep.spec.replicas or 1
                return float(replicas / self.max_replicas)
            except Exception:
                pass
        return 2.0 / self.max_replicas

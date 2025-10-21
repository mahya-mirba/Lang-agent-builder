from typing import Any

import yaml

from google.cloud import storage

from config_resolvers.base import BaseConfigResolver


GCS_SRC = 'gcs'


class GCSConfigResolver(BaseConfigResolver):
    """Config Resolver for configs in Google Cloud Storage."""

    _gcs_client: storage.Client

    def __init__(self) -> None:
        self._gcs_client = storage.Client()

    def _parse_gcs_path(self, gcs_uri: str) -> tuple[str, str]:
        """Parses the gs_uri into bucket name and blob name."""
        if not gcs_uri.startswith('gs://'):
            raise ValueError("Invalid GCS URI format. Must start with 'gs://'")

        expected_parts = 2
        parts = gcs_uri[len('gs://') :].split('/', 1)

        if len(parts) < expected_parts:
            raise ValueError(
                'Invalid GCS URI. Must include bucket and object name.'
            )

        bucket_name = parts[0]
        blob_path = parts[1]
        print(parts)
        return bucket_name, blob_path

    def _load(self, path: str) -> dict[str, Any]:
        """Load the config at path as yaml from the GCS."""
        bucket_name, blob_path = self._parse_gcs_path(path)
        bucket = self._gcs_client.bucket(bucket_name=bucket_name)
        content = bucket.blob(blob_name=blob_path).download_as_text()
        return yaml.safe_load(content)

    def load(self, path: str) -> dict[str, Any]:
        """Loads the config dict from yaml at path."""
        return self._load(path)

from config_resolvers.base import BaseConfigResolver
#from config_resolvers.gcs_yaml import GCS_SRC, GCSConfigResolver
from config_resolvers.local_yaml import LocalConfigResolver


def get_resolver(src: str) -> BaseConfigResolver:
    """Returns appropriate Resolver for src."""
    # if src == GCS_SRC:
    #     return GCSConfigResolver()
    return LocalConfigResolver()

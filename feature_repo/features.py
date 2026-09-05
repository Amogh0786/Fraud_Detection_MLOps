from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64

# Define an entity for the user
user = Entity(name="user_id", value_type=ValueType.INT64, description="User identifier")

# Point to the raw data (offline store)
user_stats_source = FileSource(
    path="data/user_stats.parquet",
    timestamp_field="event_timestamp",
)

# Define a Feature View for user historical transaction features
user_transaction_stats_view = FeatureView(
    name="user_transaction_stats",
    entities=[user],
    ttl=timedelta(days=365),
    schema=[
        Field(name="v1", dtype=Float32),
        Field(name="v2", dtype=Float32),
    ],
    online=True,
    source=user_stats_source,
    tags={},
)

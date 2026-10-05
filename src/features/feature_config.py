ML_FEATURES = [
    "event_count",
    "transaction_count",
    "customer_record_access_count",
    "download_event_count",
    "download_total",
    "records_accessed_total",
    "vip_access_count",
    "first_activity_hour",
    "last_activity_hour",
    "activity_duration_hours",
]


CONTEXT_FEATURES = [
    "failed_login_count",
    "unknown_ip_count",
    "unique_ip_count",
]


IDENTIFIER_COLUMNS = [
    "user_id",
    "date",
    "role",
]
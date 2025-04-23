#   缓存锁的过期时间，单位为秒
LOCK_EXPIRE_TIME = 600

# 更新文档启用状态缓存锁
LOCK_DOCUMENT_UPDATE_ENABLED = "lock:document:update:enabled_{document_id}"

LOCK_KEYWORD_TABLE_UPDATE_KEYWORD_TABLE = "lock:keyword_table:update:keyword_table_{dataset_id}"

LOCK_SEGMENT_UPDATE_ENABLED = "lock:segment:update:enabled_{segment_id}"

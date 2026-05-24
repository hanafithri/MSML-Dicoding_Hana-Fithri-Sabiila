from prometheus_client import start_http_server
from prometheus_client import Counter
from prometheus_client import Gauge
import psutil
import time

REQUEST_COUNT = Counter(
    'ml_requests_total',
    'Total ML Requests'
)

CPU_USAGE = Gauge(
    'ml_cpu_usage_percent',
    'CPU Usage'
)

MEMORY_USAGE = Gauge(
    'ml_memory_usage_percent',
    'Memory Usage'
)

start_http_server(8000)

while True:

    REQUEST_COUNT.inc()

    CPU_USAGE.set(psutil.cpu_percent())
    MEMORY_USAGE.set(psutil.virtual_memory().percent)

    time.sleep(5)
FROM prom/prometheus:v2.45.0
USER root

# Install envsubst
RUN apt-get update && apt-get install -y gettext-base && rm -rf /var/lib/apt/lists/*

# Fix permissions for HF (User 1000)
RUN mkdir -p /prometheus && chown -R 1000:1000 /prometheus
USER 1000

# Copy the template from the monitoring folder
COPY prometheus.yml.template /etc/prometheus/prometheus.yml.template

# Inject HF_TOKEN into the config and start Prometheus
# Note: We listen on 7860 because that's what HF expects for the UI
CMD ["/bin/sh", "-c", "\
    envsubst < /etc/prometheus/prometheus.yml.template > /etc/prometheus/prometheus.yml && \
    /bin/prometheus --config.file=/etc/prometheus/prometheus.yml \
    --storage.tsdb.path=/prometheus \
    --web.listen-address=:7860"]

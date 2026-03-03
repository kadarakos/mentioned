FROM prom/prometheus:v2.45.0
USER root

# Prometheus is Alpine-based, so use 'apk' instead of 'apt-get'
# We install 'gettext' to get the 'envsubst' command
RUN apk add --no-cache gettext

# Fix permissions for HF (User 1000)
RUN mkdir -p /prometheus && chown -R 1000:1000 /prometheus
USER 1000

# Copy the template (subtree puts this at the root)
COPY prometheus.yml.template /etc/prometheus/prometheus.yml.template

# Inject HF_TOKEN and start
CMD ["/bin/sh", "-c", "\
    envsubst < /etc/prometheus/prometheus.yml.template > /etc/prometheus/prometheus.yml && \
    /bin/prometheus --config.file=/etc/prometheus/prometheus.yml \
    --storage.tsdb.path=/prometheus \
    --web.listen-address=:7860"]

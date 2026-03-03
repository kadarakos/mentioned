FROM prom/prometheus:v2.45.0

# In the official Prometheus image, 'envsubst' is usually NOT installed,
# but 'sh' is. We will use a simple 'sed' command instead of 'envsubst' 
# to avoid needing 'apk' or 'root' access.

USER root
# Create the directory and ensure the prometheus user (built-in) owns it
RUN mkdir -p /prometheus && chown -R 65534:65534 /prometheus

# Copy the template
COPY prometheus.yml.template /etc/prometheus/prometheus.yml.template

# Switch back to the prometheus user (ID 65534 is standard for this image)
USER 65534

# Use 'sed' to replace the variable. It's pre-installed on almost every Linux.
# This avoids needing to install 'gettext' via apk.
CMD ["/bin/sh", "-c", "\
    sed \"s/\${HF_TOKEN}/$HF_TOKEN/g\" /etc/prometheus/prometheus.yml.template > /etc/prometheus/prometheus.yml && \
    /bin/prometheus --config.file=/etc/prometheus/prometheus.yml \
    --storage.tsdb.path=/prometheus \
    --web.listen-address=:7860"]


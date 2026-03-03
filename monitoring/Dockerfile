FROM prom/prometheus:v2.45.0
USER root

# Create directory with correct permissions for the prometheus user (65534)
RUN mkdir -p /prometheus && chown -R 65534:65534 /prometheus

# Copy the template
COPY prometheus.yml.template /etc/prometheus/prometheus.yml.template

# Reset the entrypoint so we can run a shell script
ENTRYPOINT []

# Switch to the prometheus user
USER 65534

# Now we can safely run /bin/sh to process our template with sed
CMD ["/bin/sh", "-c", "\
    sed \"s|\\${HF_TOKEN}|$HF_TOKEN|g\" /etc/prometheus/prometheus.yml.template > /etc/prometheus/prometheus.yml && \
    /bin/prometheus --config.file=/etc/prometheus/prometheus.yml \
    --storage.tsdb.path=/prometheus \
    --web.listen-address=:7860"]


#!/bin/sh
# Remain available over SSH; checks/training are explicit commands after login.
set -eu
if [ -z "${PUBLIC_KEY:-}" ]; then
    echo 'Set PUBLIC_KEY to your SSH public key in the Pod environment.' >&2
    exit 1
fi
umask 077
mkdir -p /root/.ssh /run/sshd
# Retain existing authorized keys when a container is restarted.
if ! grep -Fqx -- "$PUBLIC_KEY" /root/.ssh/authorized_keys 2>/dev/null; then
    printf '%s\n' "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
fi
ssh-keygen -A
# SSH sessions must find the pinned conda Python, not Ubuntu's system Python.
printf '%s\n' 'export PATH=/opt/conda/bin:$PATH' \
    'export PYTHONPATH=/opt/automanip/src' \
    'export MS_ASSET_DIR=/workspace/mani_skill_assets' > /etc/profile.d/automanip.sh
exec /usr/sbin/sshd -D -e

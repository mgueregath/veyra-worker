#!/bin/sh
# Arranque del espacio (como root): llaves, paquetes del sistema declarados, preparación del usuario y sshd.
set -e
mkdir -p /workspace /workspace/.veyra/sshd
chown 1000:3000 /workspace /workspace/.veyra
[ -f /workspace/.veyra/sshd/ssh_host_ed25519_key ] || ssh-keygen -q -t ed25519 -N "" -f /workspace/.veyra/sshd/ssh_host_ed25519_key  # la huella no cambia entre arranques
printf '%s\n' "$VEYRA_AUTHORIZED_KEYS" > /etc/ssh/authorized_keys_veyra
chmod 644 /etc/ssh/authorized_keys_veyra
# paquetes del sistema que el proyecto declara (se reinstalan en cada arranque porque el contenedor es desechable)
if [ -s /workspace/project/packages.apt ]; then
  apt-get update -qq && xargs -a /workspace/project/packages.apt apt-get install -y -qq >/workspace/.veyra/apt.log 2>&1 || echo "AVISO: falló la instalación de packages.apt (ver /workspace/.veyra/apt.log)"
fi
su veyra -s /bin/sh -c "VEYRA_PROJECT='$VEYRA_PROJECT' /usr/local/bin/veyra-bootstrap"
echo "LISTO veyra-workspace $VEYRA_PROJECT"
exec /usr/sbin/sshd -D -e -f /etc/ssh/sshd_config_veyra

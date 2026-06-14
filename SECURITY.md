# Security Policy

## Supported scope

This project is intended for local-first use. Treat memory databases, exported graphs, and activity logs as sensitive user data.

## Reporting a vulnerability

Please do **not** open a public issue for security-sensitive findings.

Use GitHub private vulnerability reporting when it is enabled for the repository. If it is not available, contact the repository maintainer through a non-public channel before sharing exploit details.

When reporting, include:

- affected version or commit
- reproduction steps
- impact description
- suggested mitigation, if known

## Safe usage guidance

- Do not commit live memory databases or backup files.
- Do not expose the HTTP daemon beyond trusted local or private-network contexts without additional authentication and transport controls.
- Review exported graph content before sharing it externally.

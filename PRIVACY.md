# Privacy Policy

Last updated: September 27, 2026

`server-memory` is a local-first MCP server for durable AI-agent memory.

## Data storage

Memory data is stored locally on the user's device in SQLite databases. Depending on how the server is used, stored data may include:

- project facts and observations
- decisions and preferences
- entity relationships and tags
- activity and session history
- file paths, configuration details, or other information explicitly provided to the memory server

`server-memory` does not operate a hosted memory service and does not automatically send stored memory data to the project maintainer.

## Data collection

`server-memory` does not include analytics, advertising, tracking, or telemetry intended to collect user data for the project maintainer.

The project maintainer does not receive or have access to users' local memory databases unless a user explicitly chooses to share them.

## Network access

The default MCP transport is local stdio.

An optional HTTP server can be enabled by the user. By default, it is intended to bind to localhost and uses bearer-token authentication. Users are responsible for securing any configuration that exposes the server beyond the local machine or trusted private environments.

## Embeddings

Optional embedding support may use locally installed or cached model files. Installing or obtaining those models or dependencies may involve third-party package or model-hosting services subject to their own privacy policies.

Core SQLite and FTS5 memory functionality does not require an external hosted database or embedding service.

## Exports and backups

Users may explicitly export or back up stored memory. These files can contain sensitive information.

Users are responsible for reviewing and protecting exported graphs, database backups, logs, and other generated files before sharing them.

## Third-party applications

`server-memory` may be used through MCP-compatible clients such as Claude Code or other applications. Those applications may process information retrieved from `server-memory` according to their own privacy policies and settings.

This privacy policy applies only to `server-memory` itself.

## Data retention and deletion

Memory remains stored locally until the user deletes it, removes individual stored records, or deletes the associated database.

`server-memory` provides tools for managing and deleting stored memory.

## Security

Memory databases should be treated as potentially sensitive data. Users should not commit live memory databases, authentication tokens, exports, or backups to public repositories.

Security issues should be reported according to the project's [Security Policy](SECURITY.md).

## Changes to this policy

This policy may be updated if the project's data handling or network behavior changes. Updates will be published in this repository.

## Contact

For privacy questions, open a repository issue that does not contain sensitive information, or contact the repository maintainer through GitHub.

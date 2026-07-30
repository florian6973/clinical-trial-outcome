# Security and data handling

The maintained Qwen workflow does not require a hosted-model API key. Do not
commit credentials, model tokens, licensed terminology, private annotations,
or unreviewed source data.

Before publishing a new dataset:

1. validate its schema and provenance;
2. confirm that no credential or local path is present;
3. apply the terminology-license boundary documented in
   `publication_data/LICENSE_AND_PROVENANCE.md`;
4. run the publication-data validators; and
5. review the staged files and Git history with a secret scanner.

Report a suspected credential or restricted-data disclosure privately to the
repository maintainers rather than opening a public issue.

# maintaining

This document is for LightGBM maintainers.

## Releasing

### Artifacts

After creating a release, run the following from the root of the repo to populate it with artifacts.

```shell
./.ci/downloads-artifacts.sh ${COMMIT_ID}
```

Where `COMMIT_ID` refers to the commit on `master` corresponding to the release.

Upload those files to the release.

```shell
gh release upload \
    --repo microsoft/LightGBM \
    "${TAG}" \
    ./release-artifacts/*
```

Where `TAG` refers to the `git` tag corresponding to the release.

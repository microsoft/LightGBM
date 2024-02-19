# contributing

LightGBM has been developed and used by many active community members.

Your help is very valuable to make it better for everyone.

## How to Contribute

- Check for the [Roadmap](https://github.com/microsoft/LightGBM/projects/1) and the [Feature Requests Hub](https://github.com/microsoft/LightGBM/issues/2302), and submit pull requests to address chosen issue. If you need development guideline, you can check the [Development Guide](https://github.com/microsoft/LightGBM/blob/master/docs/Development-Guide.rst) or directly ask us in Issues/Pull Requests.
- Contribute to the [tests](https://github.com/microsoft/LightGBM/tree/master/tests) to make it more reliable.
- Contribute to the [documentation](https://github.com/microsoft/LightGBM/tree/master/docs) to make it clearer for everyone.
- Contribute to the [examples](https://github.com/microsoft/LightGBM/tree/master/examples) to share your experience with other users.
- Add your stories and experience to [Awesome LightGBM](https://github.com/microsoft/LightGBM/blob/master/examples/README.md). If LightGBM helped you in a machine learning competition or some research application, we want to hear about it!
- [Open an issue](https://github.com/microsoft/LightGBM/issues) to report problems or recommend new features.

## Development Guide

### Linting

Every commit in the repository is tested with multiple static analyzers.

When developing locally, run some of them using `pre-commit` ([pre-commit docs](https://pre-commit.com/)).

```shell
pre-commit run --all-files
```

That command will check for some issues and automatically reformat the code.

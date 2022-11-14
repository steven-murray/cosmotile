# Cosmotile

[![PyPI](https://img.shields.io/pypi/v/cosmotile.svg)][pypi_]
[![Status](https://img.shields.io/pypi/status/cosmotile.svg)][status]
[![Python Version](https://img.shields.io/pypi/pyversions/cosmotile)][python version]
[![License](https://img.shields.io/pypi/l/cosmotile)][license]

[![Read the documentation at https://cosmotile.readthedocs.io/](https://img.shields.io/readthedocs/cosmotile/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/steven-murray/cosmotile/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/steven-murray/cosmotile/branch/main/graph/badge.svg)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pypi_]: https://pypi.org/project/cosmotile/
[status]: https://pypi.org/project/cosmotile/
[python version]: https://pypi.org/project/cosmotile
[read the docs]: https://cosmotile.readthedocs.io/
[tests]: https://github.com/steven-murray/cosmotile/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/steven-murray/cosmotile
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

_Create cosmological lightcones from coeval simulations._

This algorithm is taken from the code in https://github.com/piyanatk/cosmotile, but
is repackaged and re-tooled.

## Features

- Fast tiling of finite, periodic cosmic simulations onto arbitrary angular coordinates.
- Generate different realizations by translation and rotation.

## Installation

You can install _Cosmotile_ via [pip] from [PyPI]:

```console
$ pip install cosmotile
```

## Usage

Please see the [Command-line Reference] for details.

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license],
_Cosmotile_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [@cjolowicz]'s [Hypermodern Python Cookiecutter] template.

The algorithm used in this repository is derived from the `cosmotile` module in
https://github.com/nithyanandan/AstruUtils, which was later modularised in
https://github.com/piyanatk/cosmotile.

## Acknowledgments

If you find `cosmotile` useful in your project, please star this repository and, if
applicable, cite https://arxiv.org/abs/1708.00036.

[@cjolowicz]: https://github.com/cjolowicz
[pypi]: https://pypi.org/
[hypermodern python cookiecutter]: https://github.com/cjolowicz/cookiecutter-hypermodern-python
[file an issue]: https://github.com/steven-murray/cosmotile/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/steven-murray/cosmotile/blob/main/LICENSE
[contributor guide]: https://github.com/steven-murray/cosmotile/blob/main/CONTRIBUTING.md
[command-line reference]: https://cosmotile.readthedocs.io/en/latest/usage.html

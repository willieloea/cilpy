# cilpy: A Computational Intelligence Library for Python

`cilpy` is a Python library designed for experimenting with nature-inspired algorithms, with a special focus on dynamic and constrained optimization problems. Its modular design allows researchers and developers to easily benchmark existing algorithms or test new ones.

## Key Features

- **Modular Design**: Problems, solvers, and experiment runners are decoupled, making it easy to mix and match components.
- **Dynamic Problem Generators**: Includes the well-known Moving Peaks Benchmark (MPB) and its constrained variant (CMPB) to create challenging dynamic landscapes.
- **Rich Solver Collection**: Provides a wide array of ready-to-use solvers, including variants of PSO, DE, and GAs.
- **Experiment Runner**: A simple utility to automate running experiments and logging results to CSV.

## Installation
TODO

## Project Structure
 * `/cilpy`: The core library source code.
     * `/problem`: Contains problem generators.
     * `/solver`: Contains algorithm implementations.
     * `runner.py`: The experiment execution utility.
 * `/examples`: Standalone scripts showing how to use the library.
 * `/test`: The testing suite for the library.
 * `/dev`: Contains resources for developers.

## Contributing
Contributions are welcome! Please see the Contributing Guidelines for more details on how to set up your development environment, run tests, and submit changes.

## License
TODO

## MkDocs guide
For full documentation visit [mkdocs.org](https://www.mkdocs.org).

### Commands

* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

### Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.

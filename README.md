# The WRS Robot Planning & Control System

This is a brief guide to the WRS Robot Planning & Control System. For detailed technical descriptions and usage, go to
the [document pages](https://wanweiwei07.github.io/wrs/) hosted at [my homepage](https://wanweiwei07.github.io/).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing
purposes.

### Prerequisites

The following packages are needed to run this system

```
panda3d>=1.10.7 # vital, visualization
numpy>=1.17.4 # vital, matrix computation
pycollada>=0.7.1 # optional, required by trimesh to load dae files
opencv-python>=4.4.0.46 # optional, required by vision
opencv-contrib-python>=4.4.0.46 # optional, required by vision
scikit-learn>=0.23.2 # vital?
```

### Installing

A step by step series of examples that tell you how to get a development env running. The recommended IDE(Integrated
Development Environment) is [PyCharm](https://www.jetbrains.com/pycharm/). You can get a community version for research
purpose at [PyCharm Community Version](https://www.jetbrains.com/pycharm/download/). Other platform like Visual Studio
Code might also be helpful, although I never tested them.

Clone this repository to your local disk and open the folder as a project in your PyCharm IDE, you will see all packages
in the Project View. Their names and usage are as follows.

```
basis: Basic helper functions for math computation, data structure conversion, and trimesh processing.
drivers.devices: Wrapper for the low-level robot or sensor APIs.
drivers.rpc: Remote procedure call interfaces. To be run on remote servers.
grasping: Grasp planners.
manipulation: Stability Analyzers; Placement planners.
planning: Trajectory-level and probabilistic motion-level planners.
robotcon: Interface functions to connect and control robots.
robotsim: Robot classes are defined in this package.
vision: Utility functiosn for processing 2D and 3D vision data.
visualization: Graphics. Panda3D is the main graphics engine.
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of
conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see
the [tags on this repository](https://github.com/your/project/tags).

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

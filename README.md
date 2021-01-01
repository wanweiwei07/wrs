# The WRS Robot Planning & Control System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is a brief guide to the WRS Robot Planning & Control System. For detailed technical descriptions and usage, go to
the [document pages](https://wanweiwei07.github.io/wrs/) hosted at [my homepage](https://wanweiwei07.github.io/).

<details><summary>Japanese Translation (日本語)</summary>
このページでは，WRSロボット計画制御システムの仕組みを簡単に説明します．技術的な詳細な説明と使用方法については，
[私のホームページ](https://wanweiwei07.github.io/) にある[ドキュメントページ](https://wanweiwei07.github.io/wrs/) を参照してください．
</details>

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing
purposes.

<details><summary>Japanese Translation (日本語)</summary>
以下の手順では，開発とテストのために，プロジェクトのコピーをローカルマシンで起動して実行する目的で使用します．
</details>

### Prerequisites

The following packages are needed to run this system.

<details><summary>Japanese Translation (日本語)</summary>
まず，このシステムを実行するには以下のパッケージが必要です．
</details>

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

<details><summary>Japanese Translation (日本語)</summary>
次に，開発方法をステップバイステップで説明します．推奨するIDE(集成開発環境)は [PyCharm](https://www.jetbrains.com/pycharm/) です．
研究用に無料でコミュニティ版を入手することができます．[PyCharm Community Version](https://www.jetbrains.com/pycharm/download/)
を参照してください．また，Visual Studioのような他の開発環境も良い候補かもしれませ．
</details>

Clone this repository to your local disk and open the folder as a project in your PyCharm IDE, you will see all packages
in the Project View. Their names and usage are as follows.

<details><summary>Japanese Translation (日本語)</summary>
このリポジトリをローカルディスクにクローンし，クローンしたフォルダーをPyCharm IDEでプロジェクトとして開くと，すべてのパッケージが表示されます．
それぞれのパッケージの名称と目的は以下の通りです．
</details>

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

Besides the abovementioned packages, there is a 0000_example folder that hosts several examples. Run the following one
to examine if your the prerequisites and key packages work correctly.

<details><summary>Japanese Translation (日本語)</summary>
上記のパッケージの他，0000_example フォルダがあり，いくつかのサンプルを用意しています．その中の以下のファイルを実行して，クローンしたコード
が正しく動作するかどうかを確認してください．
</details>

```
TODO
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Thank all related students in HLab for using and suggesting to this software.

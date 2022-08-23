# Thesis preparation     <!-- omit in toc -->
In this document are contents to prepare for a thesis in the topic area of machine learning. The idea is to learn contents independently with the help of realistic tasks. (Help for self-help) 
- [Formalities](#formalities)
  - [Before writing the thesis](#before-writing-the-thesis)
- [Technical Coding Introduction](#technical-coding-introduction)
  - [Introduction to Python](#introduction-to-python)
    - [Installing Python](#installing-python)
    - [Learning Python](#learning-python)
    - [DocStrings](#docstrings)
    - [Creating a virtual environment](#creating-a-virtual-environment)
    - [PyTorch](#pytorch)
    - [Tensorboard](#tensorboard)
  - [Introduction to Visual Studio Code](#introduction-to-visual-studio-code)
    - [Installation of Visual Studio Code](#installation-of-visual-studio-code)
    - [Useful Visual Studio Code Extensions](#useful-visual-studio-code-extensions)
    - [Useful Visual Studio Code Shortcuts](#useful-visual-studio-code-shortcuts)
  - [Introduction to Github](#introduction-to-github)
    - [First Steps](#first-steps)
    - [Git in Visual Studio Code](#git-in-visual-studio-code)
    - [Policies for Commits](#policies-for-commits)
    - [Writing a README File](#writing-a-readme-file)
    - [Creating a gitignore file](#creating-a-gitignore-file)
- [Writing a thesis](#writing-a-thesis)
  - [Latex](#latex)
  - [Stil](#stil)
- [Technical Contents](#technical-contents)
  - [Invalid Action Masking](#invalid-action-masking)
  - [Q-Learning](#q-learning)
  - [Temporal-Difference Learning](#temporal-difference-learning)
  - [Decision Transformer](#decision-transformer)
  - [Actor Critic Methods](#actor-critic-methods)
  - [Markov Games](#markov-games)
  - [Variance Minimization Methods](#variance-minimization-methods)
  - [Upside Down Reinforcement Learning](#upside-down-reinforcement-learning)
- [Example Projects](#example-projects)

# Formalities
In this section there is a list of formalities for writing a thesis. 
## Before writing the thesis
- [ ] One seminar must be attended (Due to a semester abroad this can also happen in the 6th semester)
- [ ] Signing and filling in the [study plan](https://www.wim.uni-mannheim.de/studium/studienorganisation/b-sc-wirtschaftsmathematik/#c109920)
- [ ] Thesis registration  

# Technical Coding Introduction 
## Introduction to Python
In this section, the Python programming language is discussed. In particular, it shows how to [install](#installing-python) Python, how you can best [learn](#learning-python) Python on your own, and other mandatory topics such as [DocStrings](#dochstrings) and [creating a virtual environment](#creating-a-virtual-environment).
### Installing Python
On MacOs Python is already preinstalled, but this is mostly the deprecated version 2.7. You can use 
```sh
python --version
``` 
to check your Python version. Installing Python on MacOS with [homebrew](https://brew.sh/index_de) is recommended. Here are step-by-step instructions for [MacOS](https://github.com/FelixRb96/Python_kurs_RTG/blob/main/materialien/0_installation_und_entwicklungsumgebung.ipynb), [Windows](https://github.com/FelixRb96/Python_kurs_RTG/blob/main/materialien/0_installation_und_entwicklungsumgebung.ipynb) und [Linux](https://code.visualstudio.com/docs/python/python-tutorial). 
### Learning Python 
This section addresses two questions.
1. What content needs to be acquired?
2. How can content be acquired efficiently and sustainably?  
 
The first step is to learn a secure handling of Basic Python. 
This includes the following points:
1. Numeric data types
   1. `int` & `long`
   2. `float`
   3. `complex`
   4. `bool`
2. Arithmetic operators
   1. `x+y`
   2. `x-y`
   3. `x*y`
   4. `x/y`
   5. `x**y`
   6. `x//y`
   7. `-x`
3. Logical operators
   1. `==`
   2. `>=`
   3. `<=`
   4. `!=`
   5. `<`
   6. `>`
4. Sequential data types
   1. lists `list`
   2. strings `str`
   3. tuples `tupel`
5. dictionaries `dict`
6. sets `set`
7. modules
8. libraries

To acquire the contents, it is recommended to work through [this introduction](https://github.com/FelixRb96/Python_kurs_RTG/blob/main/materialien/1_python_uebersicht.ipynb) (time requiered: approx. 2h).
In a second step, a confident handling of the libraries [numpy](https://numpy.org), [pandas](https://pandas.pydata.org) and [matplotlib](https://matplotlib.org) must be acquired. The following introductions are suitable for this purpose:
1. [numpy](https://github.com/FelixRb96/Python_kurs_RTG/blob/main/materialien/2_numpy.ipynb) (time requiered: approx. 1h)
2. [pandas](https://github.com/FelixRb96/Python_kurs_RTG/blob/main/materialien/4_pandas.ipynb) (time requiered: approx. 1h)
3. [matplotlib](https://github.com/FelixRb96/Python_kurs_RTG/blob/main/materialien/3_matplotlib.ipynb) (time requiered: approx. 30min)

Tutorials offer a first overview of contents, but mostly do not promote the independent handling of Python. Therefore, the next step is to apply the learned knowledge in smaller and larger projects. For this purpose, the websites [kaggle](https://www.kaggle.com) and [HackerRank](https://www.hackerrank.com) are suitable. Participation in [STADS](https://stads.uni-mannheim.de/) projects is recommended.   
***
**Challenge**
- [ ] Achieve 50 % in Hackerrank Python Preperation 
- [ ] Participate in three kaggle challenges
- [ ] Participate in one STADS Project 
***
### DocStrings
In order for other project members to understand written code, functions and files must be documented. By default, Python uses [DocStrings](https://en.wikipedia.org/wiki/Docstring) for this purpose. In addition, [typing](https://docs.python.org/3/library/typing.html) is often used. [Visual Studio Code](#introduction-to-visual-studio-code) offers an [extension](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) for the simplified creation of DocStrings.  
*** 
**Challenge**
- [ ] Write your first DocStrings to a Python Project. 
***
### Creating a virtual environment
When creating projects, external [libraries](https://www.geeksforgeeks.org/libraries-in-python/) are usually imported into Python. These files can be several Mb's in size and should not be pushed into a Github repository, for example. In addition, version conflicts can cause problems between different Python projects if the same libraries are loaded. For this reason, a virtual environment must be created that contains the packages and configurations for a given project. In Visual Studio Code this works [like this](https://code.visualstudio.com/docs/python/environments).
A brief summary for MacOs is as follows.
1. Open a terminal within a project in Visual Studio Code
2. Create a virutal environment 
    ```sh
    python3 -m venv .venv
    ``` 
3. In the lower right corner, the question is asked whether the virtual environment should be activated. This must be confirmed. 
4. Activate the virutal environment for the terminal  
   ```sh
   source .venv/bin/active
   ```
### PyTorch
[PyTorch](https://pytorch.org) is an open source machine learning framework that accelerates the path from research prototyping to production deployment. A large number of reinforcement learning algorithms use the library. Therefore, a safe handling of the library is essential. 
First of all the 
[Introduction](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html) should be worked on. 
***
**Challenges**
- [ ] Write a costum activation function (autograd function) (for example: [sigmoid function](https://de.wikipedia.org/wiki/Sigmoidfunktion))
- [ ] Use a neural network to optimize the [classification problem](https://www.kaggle.com/code/dansbecker/classification/notebook). 
***
### Tensorboard
So-called [tensorboards](https://www.tensorflow.org/tensorboard) are used for training analysis of neural networks. The tool has the following features:
- Track and visualize metrics such as loss and accuracy
- Visualization of the model graph (ops and layer)
- View histograms of weights, biases, or other tensors that change over time
- Projecting embeddings into a lower dimensional space
- Display images, text and audio data
- Profiling TensorFlow programs

An introduction to tensorboard is [here](https://www.tensorflow.org/tensorboard/get_started) (time requiered: 30min). 
***
**Challenge**
- [ ] Work through the introduction 
- [ ] Write a tensorboard for the [example](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html).
*** 
## Introduction to Visual Studio Code
The use of Visual Studio Code has already been addressed in a [section](#git-in-visual-studio-code-vsc) on Git(-hub). Primarily, Visual Studio Code is used as an editor for programming languages such as Python, C++ or R. However, the editor has other useful features and shortcuts, which will be discussed in this section. First, we will look at the installation of Visual Studio Code.
### Installation of Visual Studio Code
Visual Studio Code ca be downloaded [here](https://code.visualstudio.com/download). It is recommended to use a step by step instruction for [MacOS](https://code.visualstudio.com/docs/setup/mac), [Linux](https://code.visualstudio.com/docs/setup/linux) or [Microsoft](https://code.visualstudio.com/docs/setup/windows). 
### Useful Visual Studio Code Extensions 
Visual Studio Code has a number of useful extensions that simplify programming. For example, the [Python](https://code.visualstudio.com/docs/languages/python) extension offers intelligent auto-completion, error detection and a debugger tool.
Other useful extensions are:
1. **Latex Extension**: Visual Studio Code can be used as an editor for [Latex](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop). Advantages are code highlighting and autocompletion. 
2. **Auto formatter**: Python code, latex code etc. can be put into a readable (best practice) format by auto-formatting. Important: Only formatted code is pushed. Otherwise there will be big trouble from MK. 
3. **Markdown**: The [Markdown Extension](https://code.visualstudio.com/docs/languages/markdown) allows the simplified creation of `.md` files. 
4. **DocStrings**: A program must be documented. This is important for external as well as internal use. So-called [DocStrings](https://en.wikipedia.org/wiki/Docstring) are used for this purpose. Visual Studio Code offers an [Extension](https://towardsdatascience.com/3-easy-steps-to-folding-docstrings-in-vscode-fbb64573611b) for Python, with the help of which DocStrings can be written more easily. 
### Useful Visual Studio Code Shortcuts
To program quickly and efficiently, shortcuts are indispensable. For MacOs the following shortcuts must be known:
1. &#x2318; + D
2.  &#x2318; + S
3.  &#x2318; + Z
4.  &#x2318; + &#x21E7; + P
5.  &#x2318; + P
6.  &#x2318; + &#8594; 
7.  &#x2318; + &#8592;
8.  &#x2318; + &#x2325; + &#8595;

A list of all shortcuts for Visual Studio Code kann be found by using the shortcut: &#x2318; + &#x21E7; + P  ```>Prefercences: Open Keyboard Shortcuts ```  

## Introduction to Github
This section is intended to learn basic knowledge of Git(-hub). [Git](https://de.wikipedia.org/wiki/Git) is a free software for distributed version management of files. [GitHub](https://de.wikipedia.org/wiki/GitHub) is a network-based version management service for software development projects. It was named after the version control system Git. The company GitHub, Inc. is based in San Francisco in the USA. As of December 26, 2018, the company is part of Microsoft. Similar services are GitLab, Bitbucket and Gitee.
### First Steps
For the introduction, a first repository is to be created and basic commands are to be learned. To do this, work through the [example](https://product.hubspot.com/blog/git-and-github-tutorial-for-beginners) (time required: approx. 1h). 
Goals of the task/questions to be answered:
1. Creation of an account on Github
2. First commit, push and pull of a repository
3. What is the purpose of the following commands? 
   - ```sh 
      git init 
      ```
   - ```sh
      git status 
      ```
   - ```sh
      git add 
      ```
   - ```sh
      git commit 
      ```
   - ```sh
      git branch 
      ```
   - ```sh
      git remote 
      ```
   - ```sh
      git push 
      ```
   - ```sh
      git merge 
      ```
   - ```sh
      git fetch 
      ```
   -   ```sh
        git pull
        ```
   -   ```sh
        git checkout
        ```
   -   ```
        git diff
        ```
4. What is the difference between a 'push' and a 'commit'?
5. What are 'branches' used for?
### Git in Visual Studio Code
[Visual Studio Code](https://de.wikipedia.org/wiki/Visual_Studio_Code) is a free source code editor from Microsoft. Visual Studio Code is available cross-platform for the operating systems Windows, macOS and Linux and is based on the Electron framework and enables syntax highlighting, code folding, debugging, auto-completion and version management, among other things. In the meantime, the editor has established itself at most companies and also by Adaconis (people who work at Adacon). Visual Studio Code has a plug-in which facilitates the handling of Git(-hub). To develop a better feeling for Git(-hub) it is worth to use the extension. For more complicated projects (the extension has its limits), using it via the terminal is strongly recommended.
The task is to push the repository created in the previous section to a remote repository using Visual Studio Code. Instructions can be found [here](https://code.visualstudio.com/docs/editor/github) (time required: approx. 30 min).
### Policies for Commits
For a structured and successful project work to be possible, commits must be designed sensibly (wip, add 'file' is not a good style and will take revenge in the course of the project) and branches must be used appropriate. Each project agrees on its own rules beforehand. An example for such rules can be found [here](https://www.youtube.com/watch?v=Uszj_k0DGsg).
### Writing a README File
A README.md file must be created so that members outside of a project are informed about the project progress or about the use and contents of an projet. A file with the extension .md is a called Markdown file. [Markdown](https://de.wikipedia.org/wiki/Markdown) or a Markdown-like syntax is mainly used on developer platforms with a more tech-savvy audience such as GitHub, Stack Overflow or the blogging platform Ghost. 
A README file is subject to a certain structure and must answer certain questions. The most important points are:
1. **Project Title**: What is the goal of the project 
2. **Project describtion**: 
   1. What does your application?
   2. Why are you using your application?
   3. What challanges do you face? 
3. **Table of Contents**
4. **Installation guide**
   1. How to install the application? 
   2. How to use the main functionalities of the project?
5. **Credits**: List of all contributors
6. **License** 
    
A detailed description of the individual points can be found [here](https://www.freecodecamp.org/news/how-to-write-a-good-readme-file/). An overview of the creation of a Markdown file/the most important commands can be found [here](https://www.markdownguide.org/cheat-sheet/). [Emojis](https://gist.github.com/rxaviers/7360908) and [keyboard commands](https://gist.github.com/zmwangx/10571883) can also be added in  a markdown file.
### Creating a gitignore file 
One of the main reasons for using Git(-hub) is that it greatly simplifies the collaboration of multiple developers on a project. One hurdle, however, is files that exist only to configure a single developer's project environment. These are typically created by development environments like Xcode or Android Studio, and they store things like the last file opened or the configuration of individual panels (subwindows) in the development environment. So that these settings are not taken over by other developers but each can keep his own settings, there is the possibility in Git to exclude certain files from commits in general.
Ignoring files takes place in Git through a simple listing in a text file called ```.gitignore```.
For a python project the ```.gitignore``` file must include [these](https://stackoverflow.com/questions/3719243/best-practices-for-adding-gitignore-file-for-python-projects) items. Important: Depending on the operating system (Linux, MacOS, Windows) the file (contents of the file) looks different. 

# Writing a thesis
## Latex
The thesis will be written in [Latex](https://de.wikipedia.org/wiki/LaTeX). [Visual Studio Code](#useful-visual-studio-code-extensions) is recommended as the editor for Latex. An introduction to Latex is [here](https://www.youtube.com/watch?v=-HvRvBjBAvg). Using macros and including libraries, the format of a Latex file can be customized. [Here](https://www.youtube.com/watch?v=331YxgOJUGw) are a few examples. 
***
**Challenge**
- [ ] Open a Github repository of a lecture with friends and use Latex together to text difficult proofs of the lecture.   
***
## Stil 
# Technical Contents
In this chapter interesting topics from machine learning are presented. 
## [Invalid Action Masking](https://arxiv.org/abs/2006.14171) 
## [Q-Learning](https://www.researchgate.net/publication/220344150_Technical_Note_Q-Learning)
## [Temporal-Difference Learning](https://link.springer.com/article/10.1007/BF00115009)
## Decision Transformer
## [Actor Critic Methods](Papers/Actor_Critic/Natural_Actor_critic.pdf)
## Markov Games
## Variance Minimization Methods 
## Upside Down Reinforcement Learning 
# Example Projects 





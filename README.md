# PyTorch Base

## A Template Repository for PyTorch-Based Projects

Any PyTorch project is mostly built on template code &mdash; code which does not change, almost no matter the field, data, model, etc. For that reason, it is convenient to have a template which takes care of all the boilerplate code. Moreover, depending on the model and dataset size, you might want to run the code on multiple GPUs. Other times, for example in development, you might want to run the same code with a smaller model and dataset on CPU or a single GPU. To satisfy my own needs of a flexible code base meeting the above requirements, I have created this repository to act as a template for future projects. 

<u>**This repository is under development!**</u>

## Some Features and Design Choices

I wanted this repository to be simple to integrate into projects with diverse application areas. I also wanted the *same* code to be run-able on CPU and GPU with minimal difference in UI &mdash; I often do development and method tests on smaller datasets and architectures, and quickly want to transition to larger tests. 

The above is nothing unique; most projects I see have the above features. However, in my experience, the ``main.py`` file is convoluted in many projects &mdash; it often contains 4 or even 5 lines of indentation to handle various runtime options, GPU/CUDA settings, and so on. Moreover, since the main file is often long, it is often hard to get an overview of the core algorithm being developed. Simply put, it is often being hidden by boilerplate code, logging, and CUDA semantics. 
The above is nothing unique; most projects I see have the above features. However, in my experience, the ``main.py`` file is convoluted in many projects &mdash; it often contains 4 or even 5 lines of indentation to handle various runtime options, GPU/CUDA settings, and so on. Moreover, since the main file is often long, it is often hard to get an overview of the core algorithm being developed. Simply put, it is often hidden by boilerplate code, logging, and CUDA semantics. 

My design philosophy is highly influenced by the above opinions. I have therefore designed the files in the following manner:

- ``main.py``: Handles imports, CPU/GPU semantics, logging, checkpointing, and main training loop. No core logic is put here, all the important bits are imported via *getters* from other files. 
- ``utils.py``: Parses arguments, and through those, defines the *getters*. Also contains the training and evaluation loops, which can easily be modified later by the parsed arguments. 
- ``models.py``: Define the models to be imported in ``utils.py``.
- ``datasets.py``: Imports train, validation, and test splits from specified datasets, to be imported in ``utils.py``.

The default behaviour is classification on images. However, as we shall soon see, the core logic is largely agnostic to this design choice.

## How To Modify

In most projects, you would most likely want to do something beyond classification using the pre-built datasets and models I have built in. Here I describe a typical workflow to do that. 

Assume you want to import a new dataset of some modality, train a new architecture, with a new loss. The following steps could be involved:

1. Check the command-line arguments in ``parse_arguments()`` in ``utils.py``. 
    - You probably have to add/remove/change arguments to accommodate new parameters and hyperparameters. 
    - If you change the default arguments, you probably have to change the corresponding *getters* to avoid errors.
2. Define the model in ``models.py`` and import the dataset in ``datasets.py``. 
3. Update the *getters* in ``utils.py``. 
    - In this example, for a new model, dataset, and presumably loss function, the corresponding *getters* have to be augmented to avoid throwing errors. 
    - Make sure to import all relevant things from ``models.py``and ``datasets.py``.
4. Check if the standard training and evaluation routines in ``utils.py`` are sufficient. 
    - Consider changing these in-place, or branching these out into different functions, if your application requires it. 
    - You might have to change the logging routines in ``utils.py``, and possibly how these are handled in in ``main.py``.
5. Run!

Your mileage may of course vary, but hopefully this gives an indication of the complexity of modifying the base code.

## Usage

The default behaviour is to train vision nets for classification with cross-entropy loss. For examples of common run settings, see the ``.sh``files in the ``example_runs`` folder.

To see all run options, run ``main.py`` with the ``--help`` flag, or check the source code in ``parse_arguments()`` in ``utils.py``.

## Why Choose This Over ``PyTorch-Lightning``?

I know very little about ``Lightning``, but to my understanding, this repository would have a similar function as a basic ``Lightning`` script. If you feel more comfortable with that, go ahead. 

(The reason I created this repository is that I do not feel comfortable with ``Lightning``, and felt it would be equally cumbersome to write this as it would be to learn Lightning. Your time-reward calculus might be different.)
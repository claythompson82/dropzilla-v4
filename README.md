Dropzilla v4: An Institutional-Grade Intraday Signal Engine
This repository contains the source code for Dropzilla v4, a complete redesign of the intraday short-biased signal generation system. This version is architected from the ground up to be modular, scalable, and methodologically sound, addressing the core limitations of previous iterations.

Project Scope & Purpose
The primary goal of Dropzilla v4 is to identify high-conviction, short-biased trading opportunities in liquid US equities on an intraday basis (minutes-to-hours horizon).

This system is built on several core principles:

Methodological Rigor: All backtesting and optimization uses a leak-free, time-series-aware Walk-Forward Analysis framework.

Adaptive Modeling: The system is designed to combat concept drift by using rolling data windows and explicit market regime detection.

Principled Conviction: Signal confidence is not a raw probability but a multi-factor score derived from model output, signal stability, market context, and volume confirmation.

Engineering Excellence: The codebase is a clean, testable, and maintainable Python package to facilitate rapid and reliable development.

Installation (WSL2 / Ubuntu)
These instructions assume you are using Ubuntu within a WSL2 environment on Windows 11.

Clone the Repository:

git clone <your-private-repo-url>
cd dropzilla-v4

Create and Activate a Python Virtual Environment:
It is strongly recommended to use a virtual environment to manage project dependencies.

python3 -m venv .venv
source .venv/bin/activate

Install Dependencies:
Install all required packages from the requirements.txt file.

pip install -r requirements.txt

Install Dropzilla in Editable Mode:
Installing the package in editable mode (-e) allows you to make changes to the source code and have them immediately reflected without needing to reinstall.

pip install -e .

The system is now installed and ready for development or execution.

Running Tests
This project uses pytest for testing. The initial tests are simple placeholders to ensure the testing framework and CI pipeline are functional.

To run all tests, execute the following command from the root directory of the project:

pytest

You should see all tests passing.

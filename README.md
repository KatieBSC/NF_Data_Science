# NF DataScience Course 1

In this repository you will find example code alongside the topics of the course. It is structured by the topics discussed within the course.

- intro: Notebooks loading basic python packages we used and contains simple examples
- descriptive analysis: Notebooks created to analyze and visualize data using descriptive methods

## Setup

```sh
git clone https://github.com/maddosz12/NF_DS_1_course_material.git
```

```sh
conda env create --file environment.yml
source activate neue-fische
```

## Usage

```sh
conda env update --file environment.yml
source activate neue-fische
jupyter notebook
```

## Tests

```sh
pytest # tests
```

```sh
flake8 # codestyle
```

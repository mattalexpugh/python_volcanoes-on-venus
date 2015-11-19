# Volcanoes on Venus

----

This is a Python library for interfacing with the [Volcanoes on Venus data-set](https://kdd.ics.uci.edu/databases/volcanoes/volcanoes.html), that I built for a project in [IRDS](http://www.inf.ed.ac.uk/teaching/courses/irds/) at the University of Edinburgh.

The original project provides a MATLAB interfaces to obtain the image fragments from the custom format provided, but it's 2015. Who still uses MATLAB?

## Setup

A `config.ini.sample` is given, copy this to `config.ini` and update the path field within to wherever you've put the data-set linked above. 

I haven't made a `.egg` setup etc. as, simply, I didn't need it. I'm happy to receive pull-requests for this.

## Usage

~~~python
from pyvov import ChipsIndex

ci = ChipsIndex()

all_experiments = ci.experiments()

training_split = ci.training_split_for(EXP_NAME)
testing_split = ci.testing_split_for(EXP_NAME)
all = ci.all_for_exp(EXP_NAME)
labels = ci.labels_for(EXP_NAME)

~~~


## Requirements

This was written for Python 2.7.6, but should work for 2.7.x, the only third-party library needed is NumPy:

~~~bash
pip install --user numpy
~~~
	
## License

This is distributed under the [MIT license](https://opensource.org/licenses/MIT).
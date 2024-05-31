# quantum-channel-approximation
Python code for simulating quantum channels using quantum computers.

The required python packages can be installed with the `requirements.txt` file (note the specific numpy version).
To install all packages at once you can use
> `python -m pip install -r requirements.txt`

(I did this in a virtual environment because of some issues with versions of already installed
packages and versions that were needed to run the code from which this repo was forked.)

To run the files in the results folder you first have to `pip install` my code,
(this is done via the `.toml` file, I recommend using the flag `-e` so you 
don't have to reinstall after making changes to the library everytime)
> `python -m pip install -e .`

The results folder has some notebooks where you can see some examples of how the code can be used
to approximate quantum channels.
There is also some documentation available in the form of doc strings
which has been rendered using `sphinx` for your convenience, see `docs/_build/html.index.html`

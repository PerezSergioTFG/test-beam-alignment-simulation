#Monte Carlo simulation of alignment effects in test beam experiments
The only requirements needed to run the script are to install Python and the libraries Click, Matplotlib, Numpy and Seaborn. 
The recommended versions are Click 8.1.3, Matplotlib 3.7.1, Numpy 1.23.5 and Seaborn 0.12.2.

To run the script:
```console
python "test_beam_alignment_simulation.py"
```

We built a CLI so that it is possible to change the parameters of the simulation with the commmand line.
For example to make it run 100 events you should use:
```console
python "test_beam_alignment_simulation.py" -n 100
```

For more information about the CLI:
```console
python "test_beam_alignment_simulation.py" --help
```

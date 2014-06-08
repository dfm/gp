For this lab, we'll be running an IPython notebook in an interactive job on
the Lion-XV cluster. Most of the instructions are there but we have to play a
little bit of SSH magic to get this to work properly. In all of the steps
below, change `PORT` to your unique assigned port number.

First, you need to SSH into the login node and create an SSH tunnel to forward
the correct port back to your local machine. To do this, run the following
command from your local terminal (don't forget to substitute the correct port
number):

```
# On your local machine:
ssh -L PORT:localhost:PORT lionxv.rcc.psu.edu
```

On the cluster, `cd` into your work directory and grab the code that you'll
need from the [GitHub repository](https://github.com/dfm/gp):

```
# On the cluster login node:
cd work
git clone https://github.com/dfm/gp
cd gp
```

Then we'll start up an interactive job using PBS (asking for 2 hours just to
be safe):

```
# On the cluster login node:
qsub -I -l nodes=1:ppn=1 -l walltime=2:00:00 -q WHAT_QUEUE_DO_WE_USE
```

Once that job starts up, load the correct Python module:

```
# In the interactive job:
module load python/2.7.3
cd $PBS_O_WORKDIR
```

Start a "reverse" SSH tunnel (just trust me on this one; remember to change
the port number):

```
# In the interactive job:
ssh -f -N -R PORT:127.0.0.1:PORT lionxv.rcc.psu.edu
```

And start up IPython (the port... remember):

```
ipython notebook --no-browser --matplotlib=inline --port=PORT
```

Finally, on your local machine, open up a web browser and point it at the URL:
http://localhost:PORT (replacing `PORT` with the right number).

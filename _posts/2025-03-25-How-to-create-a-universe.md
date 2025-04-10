---
layout: post
title: "How to Create a Universe: A Simple Starting Point of Running a Cosmological Simulation on your PC"
date: 2025-03-25 15:53 +0800qin
tags: [astrophysics, N-body Simulation, Python]
---

*This little project originates from one of my master courses.
*The author of this post is just a student who are interested in astrophysics, thus if you have any advise or criticism please feel free to [let me know]("tianzeng7-c@my.cityu.edu.hk" <tianzeng7-c@my.cityu.edu.hk>).

*The example scripts have been uploaded to [Github](https://www.istarshooter.com/user/34977).

Have you ever imagined that someday create a universe of your own in your childhood? Have you ever heared that some scientists on our little blue planet are working on simulating how the universe forms and evolves? And have you realized that you could try to run your own simulation on your PC? This post is actually a guide of making a very very simple cosmological simulation code.

**Step 1: Physical Scenario**

Modern cosmology theory claims that:

The early universe used to have a nearly uniform matter distribution, with very small density fluctuations. Over hundreds of millions of years, gravity amplified these fluctuations: denser regions attracted more matter, while emptier regions lost their material. Dark matter and other components first formed web-like filaments and dense halos. Within these halos, gas collapsed to create stars and galaxies, eventually building today’s cosmic structures – galaxy clusters connected by filaments, separated by vast voids. This process continues as gravity reshapes the universe’s large-scale structure. This is the big picture of our simple simulation.

<div style="text-align: center;">
  <img src="/assets/images/boxImage_TNG300-1_gas-coldens_thinSlice_1000.jpg" alt="Cosmic web" width="90%" />
</div>
*This is how the large-scale stucture looks like. The figure is from [IllustrisTNG project](https://www.tng-project.org/media/).
 
However, as a very very simple one, our simulation won't consider all of these elements. We have some assumptions to simplify the situation.

***Assumption 1:** no inflation here. In order to reduce the computational cost, we do not consider the cosmic inflation. By the way, it could be experesssed as that our result is just some certain region of our universe.*

***Assumption 2:** only dark matter. **This is one of the core assumptions of this simulation.** On the one hand, dark matter is usually considered as some kind of mysterious collisionless, purely-gravitational particles, which means that we could save many computational resources: no collision, no Navier-Stokes equation, no chemical evolution... On the other hand, baryonic matters are not that important: although they dominate the feedback processes, they just take a very small fraction of our universe.*

***Assumption 3:** this is a 2D universe. **This is one of the core assumptions of this simulation** as well. The reason is quite clear: more dimentions need more data, more storage, more computation...*

***Assumption 4:** we adopt newtonian gravitation completely. It is very understandable as well: save resource and is still precise enough due to almost no relativistic process here(we even abandoned Friedmann equations!).*

**Step 2: Simulate your own universe with code**

We have reviewed the basic physics of our simulation. Now we can try to write codes. The tool is Python. Let's start to complete it one part by one part. I use Jupyter, so we could discuss by blocks.

***Block 1: some initial setup works***

```python
## Initial conditions
# ======================

import numba
import numpy as np
import matplotlib.pyplot as plt

# Setup the number of CPU cores of computation
numba.set_num_threads(16)

# Parameters
n = 120            # number of particles is n*n
L = 100.0         # Length of this universe
sigma = 1e-2 * L  # factor of initial perturbation
r_cut = 20.0       # cut radius
softening = 0.4    # avoiding infinite gravitation
dt = 0.01          # time step
steps = 1000        # total steps
grid_size = 500  # girds of figures autosaved
v_factor = 1e0  # factor for Gaussian velocity field

# Create center coordinates of each grid
grid = np.linspace(0, L, n, endpoint=False) + 0.5 * L/n
x_centers, y_centers = np.meshgrid(grid, grid)

# Create Guassian random displacement field
np.random.seed(42)
dx = np.random.normal(0, sigma/np.sqrt(2), (n, n))
dy = np.random.normal(0, sigma/np.sqrt(2), (n, n))

# Periodic packing function
def periodic_wrap(pos, L):
    return pos % L

# Initial velocity and displacement fields
x_init = periodic_wrap(x_centers + dx, L)
y_init = periodic_wrap(y_centers + dy, L)

vx = v_factor * dx / sigma
vy = v_factor * dy / sigma

particles = np.stack([x_init.ravel(), y_init.ravel(), vx.ravel(), vy.ravel()], axis=1)

# Visulization
plt.figure(figsize=(20,10))
plt.subplot(121)
plt.scatter(particles[:,0], particles[:,1], s=0.05, c='b')
plt.title("Initial Positions")
plt.xlabel("x "), plt.ylabel("y ")

plt.subplot(122)
plt.quiver(particles[:,0], particles[:,1], particles[:,2], particles[:,3], scale=50)
plt.title("Initial Velocity Field")
plt.tight_layout()
plt.show()


```

We begin by defining a 2D "universe box" where particles initially follow a near-uniform grid with small perturbations. This mimics the early universe's density fluctuations that later grow into galaxies under gravity. And then we output a plot to have a visual expression of the initial perturbation field.

We can simply discuss the implementation of initial perturbation. The initial density fluctuations in the universe originate from quantum fluctuations during cosmic inflation. These perturbations follow a Gaussian random field with a nearly scale-invariant power spectrum (predicted by ΛCDM cosmology). In our 2D simulation:

· Gaussian displacements (dx, dy) model these primordial density variations.

· Amplitude control: sigma sets the perturbation magnitude, analogous to the primordial fluctuation amplitude. Smaller sigma → Weaker perturbations → Slower structure formation/ larger region that the simulation corresponds to.

· Scale dependence: The /np.sqrt(2) normalization ensures proper variance distribution in 2D Cartesian coordinates.

· The initial velocity field (vx, vy) is scaled from displacements. This implements the Zel'dovich approximation – a first-order Lagrangian perturbation theory that relates initial velocities to density gradients.

· You should know that *np.random.seed(42)* is a pseudo-random method which is good for testing any ideas such as comparing results of different parameter values.

OK, let's see the output:
<div style="text-align: center;">
  <img src="/assets/images/output1.png" alt="output 1" width="90%" />
</div>

<div style="text-align: left;">
  <img src="/assets/images/output2.png" alt="output 1-heatmap" width="55%" />
</div>

***Block 2: computing the gravitation***







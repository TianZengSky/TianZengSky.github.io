---
layout: post
title: "How to Create a Universe: A Simple Starting Point of Running a Cosmological Simulation on your PC"
date: 2025-03-25 15:53 +0800qin
tags: [astrophysics, N-body Simulation]
---

*The author of this post is just a student who are interested in astrophysics, thus if you have any advise or criticism please feel free to [let me know]("tianzeng7-c@my.cityu.edu.hk" <tianzeng7-c@my.cityu.edu.hk>).

*The example scripts have been uploaded to [Github](https://www.istarshooter.com/user/34977).

Have you ever imagined that someday create a universe of your own in your childhood? Have you ever heared that some scientists on our little blue planet are working on simulating how the universe forms and evolves? And have you realized that you could try to run your own simulation on your PC? This post is actually a guide of making a very very simple cosmological simulation code.

**Step1: Physical Scenario**

Modern cosmology theory claims that:

The early universe used to have a nearly uniform matter distribution, with very small density fluctuations. Over hundreds of millions of years, gravity amplified these fluctuations: denser regions attracted more matter, while emptier regions lost their material. Dark matter and other components first formed web-like filaments and dense halos. Within these halos, gas collapsed to create stars and galaxies, eventually building today’s cosmic structures – galaxy clusters connected by filaments, separated by vast voids. This process continues as gravity reshapes the universe’s large-scale structure. This is the big picture of our simple simulation.

<div style="text-align: center;">
  <img src="/assets/images/boxImage_TNG300-1_gas-coldens_thinSlice_1000.jpg" alt="Cosmic web" width="60%" />
</div>
*This is how the large-scale stucture looks like. The figure is from [IllustrisTNG project](https://www.tng-project.org/media/).
 
However, as a very very simple one, our simulation won't consider all of these elements. We have some assumptions to simplify the situation.

***Assumption 1:** no inflation here. In order to reduce the computational cost, we do not consider the cosmic inflation. By the way, it could be experesssed as that our result is just some certain region of our universe.*

***Assumption 2:** Only dark matter. **This is the core assumption of this simulation.** On the one hand, dark matter is usually considered as some kind of mysterious collisionless, purely-gravitational particles, which means that we could save many computational resources: no collision, no Navier-Stokes equation, no chemical evolution... On the other hand, bayonic matters are not that important: although they dominate the feedback processes, the just take a very small fraction of our universe.*




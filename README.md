# Parallel decomposition of the 2D heat diffusion problem

This repo is our work for an exercise for the class "Parallel Systems" in the Department of Informatics & Telecommunications in UoA. The following is a small summary of our work, our full report is located in Readme.pdf in Greek. Some tools/scripts used were excluded or lost and the project is no longer maintained, only serves as a sample of our work. However, don't hesitate to reach us through this repo's issues or contact us at samouch@gmail.com for any inquiries.

## Abstract

In this exercise, we were tasked to find and implement an efficient parallel decomposition of the 2D heat diffusion problem. The "2D heat diffusion problem" is the problem to calculate the diffusion of heat on a surface, given that:

- The surface is considered a 2D shape with no depth, which would make the analysis more complicated.
- The surface discretely segmented in small cells. Therefore, a surface of dimensions 1024 × 1024 could be represented by a 1024 × 1024 array.
- The surface has a pre-heated starting state, which commonly has the surface heated in the center and a gradient reduction of the heat moving further from the center. However, the starting state may follow another pattern or it may follow no pattern at all, random cells can have random heat.
- The perimeter of the surface is axiomatically on zero degree temperature and does not get heated.
- The temperature of every cell changes in every time step as the weighted average of the temperature of itself and its neighbors 

## Algorithmic Approach

We were given a program that implements a rather inefficient decomposition of the problem. The provided example program divided the surface in columns, while our approach was to divide the surface in a grid of sub-surfaces. Using the "Foster design methodology" we were able to prove that this decomposition was more efficient. 

In every timestep, all cells are updated once, according to its and its neighbors current value. This causes the problem of how sub-surfaces boundary cells could get updated if their value is dependent of the value of boundary cells in other subsurfaces. For this reason, the subsurfaces have to "communicate" and exchange the current status of their boundary elements. 

## Technology Used

We have developed three type of programs as implementations to the above:

-  A program using the MPI standard. The program divides the sub-surfaces in processes that utilize the MPI standard for communication. Boundary cells are exchanged between neighbor process using message passing. Various tests were made to make the communication as efficient as possible, using asynchronous communication, prepared messages etc.
- Segmentation of the surface in sub-surfaces has limits in terms of efficiency. Therefore, using the above implementation of MPI and using openMP in every MPI process, we implemented fine-grain parallelization for the part of cell updating. Again we experimented with different types of loops segmentation (dynamic, static etc), techniques to reduce cache misses since we dealt with shared memory etc.
- The third type of program was independent of the other two and utilized the parallelization potential of GPUs. It was implemented in CUDA. It uses the same decomposition discussed earlier for the GPU's blocks and shared memory between blocks to implement the boundary points exchange.

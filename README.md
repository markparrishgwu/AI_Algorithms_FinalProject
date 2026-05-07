# MPPI For The Roboracer Platform

*Mark Parrish — AI Algorithms S26 — 5/7/2026*

---

## Running the Simulation

Clone the repository and install dependencies:

```bash
git clone https://github.com/markparrishgwu/AI_Algorithms_FinalProject.git
cd AI_Algorithms_FinalProject
pip install pygame numpy
```

This program was designed and tested on Python 3.13.13

Ensure `raceline.csv` is in the same directory as `mppi.py`, then run:

```bash
python mppi.py 
```

The simulation window will open showing the raceline in yellow and the car in red. The MPPI predicted trajectory is drawn in cyan.

---

## Problem Statement

The goal of this project is to design and implement a stochastic MPC algorithm (MPPI) for use on the RoboRacer platform that will allow the autonomous vehicle to navigate a closed circuit at high speed while maintaining vehicle control. The controller will approximately solve a constrained optimization problem over a fixed horizon (defined by compute limitations) based on the bicycle model of equations in order to produce individual timestep actuator outputs (throttle/brake application, steering angle). This algorithm will eventually be used to generate training data for a policy neural net that will approximate MPPI with a longer horizon than is possible to compute in real time on the vehicle's hardware.

High speed autonomous vehicle control is an extremely important problem to be focusing on at the moment with the current zeitgeist of self-driving cars. In the real world, even a heavily simplified and discretized state space quickly balloons past what any current hardware would be able to search in real time. As such, algorithms like MPC that allow autonomous vehicles to follow calculated paths with a mathematically provable optimality (over the defined time horizon) are a clear choice when it comes to safety and performance critical applications. Furthermore, in the specific application of racing, there are many competing constraints that make this, structurally, an interesting SAT problem in itself. Calculating the best way to advance forward on the track while simultaneously maintaining traction, preparing for future corners by shifting track position, attempting to overtake opponents ahead while defending those behind, etc. leads to interesting behaviors.

---

## Related Work

### Error based approaches

Outside of known driving dynamics controllers like PID, 2dof PID, and Pure Pursuit which only take into account the current state of error from the desired path with little regard to how the current action will translate into future time steps, MPC and its variants are defined by that lookahead. MPPI in particular lends itself to limited on-board compute application by its stochastic nature. Even so, there is still a high compute cost that rises exponentially with the predefined time horizon of calculation, and therefore the problem is limited in several areas like sensor input, hardware fidelity, and available hardware for algorithm computation.

### MPC vs. MPPI

The algorithm chosen for this implementation, a variant of Model Predictive Control known as Model Predictive Path Integral, is an interesting member of the MPC family that uses stochastic sampling to derive a nearly-optimal immediate control input for a vehicle traveling along a path. MPPI is notably different from other MPC variants due to its stochastic nature. Basic MPC uses carefully defined linear systems of equations that must be invertible and solvable in order to mathematically determine a provably optimal control input for the current timestep. MPPI, on the other hand, uses randomly generated lookahead simulations (known as rollouts) and a cost function in order to come up with a solution that is nearly as good in practice, is far easier to setup with any given model function, and is much less computationally expensive than MPC with a solver, allowing for longer horizons with the same compute and better realtime performance.

The MPC vs. MPPI debate is worth expanding on a bit further. MPC is often chosen for hypercritical applications because of its provably optimal nature. However, when zooming back out into the real world, the physical situations that MPC is modeling is rarely as cooperative as the algorithm requires, leading to necessary abstractions to the equations to make them smooth and easy to work with. This abstraction often compromises the fidelity that the original programmer was looking for in the first place. MPPI, while not provably optimal, allows for an engineer to choose just about any continuous function that accurately models the situation at hand, and uses sampling to explore it rather than solving. The difference between these approaches tends to 'come out in the wash' and the algorithms are generally quite comparable in terms of performance, while MPPI usually maintains a smaller computational expense for a horizon of the same length.

---

## State Space, Actions, Transitions, and Observations

*Adapted from research notes*

### State Space

$$\mathbf{x} = [x,\ y,\ \psi,\ v]$$

where $x, y$ are the global position of the car in metres, $\psi \in (-\pi, \pi]$ is the heading angle in radians measured from the global X-axis, and $v \in [0, 15]$ m/s is the longitudinal speed. The speed limits are roughly equal to the real RoboRacer platform.

### Actions

$$\mathbf{u} = [\delta,\ a]$$

where $\delta \in [-0.44, 0.44]$ rad is the front wheel steering angle and $a \in [-6.15, 6.15]$ m/s² is the longitudinal acceleration, both derived from the physical limits of the RoboRacer platform.

### Transitions

State transitions (for MPPI and the larger simulation) follow the kinematic bicycle model with integration at timestep $\Delta t$:

$$x_{t+1} = x_t + v_t \cos(\psi_t)\, \Delta t$$

$$y_{t+1} = y_t + v_t \sin(\psi_t)\, \Delta t$$

$$\psi_{t+1} = \psi_t + \frac{v_t}{L} \tan(\delta_t)\, \Delta t$$

$$v_{t+1} = \text{clip}(v_t + a_t\, \Delta t,\ 0,\ v_{\max})$$

where $L = 0.324$ m is the wheelbase of the platform.

Interesting future work would be to upgrade this model with dynamic suspension and tire traction (quarter-car suspension model and Pacejka tire model) 

### Observations

At each timestep the controller observes the cross-track error $\text{CTE}_t = \lVert \mathbf{p}_t - \mathbf{p}^{\text{ref}}_t \rVert_2$, the heading error $\psi^{\text{err}}_t = \psi^{\text{ref}}_t - \psi_t$, and the speed error $v^{\text{err}}_t = v^{\text{ref}}_t - v_t$, where the reference values are retrieved by projecting the vehicle's arc-length position onto the precomputed raceline.

In real life, these factors would be calculated via sensor data/odometry such as LiDAR/SLAM and depth sensing cameras. Simulating those was outside the scope of this project.

---

## Solution Method

### How MPPI actually works

Imagine that you are driving a car along a straight road that is coming up to a left turn. It is quite dark and you are only able to see some fixed distance ahead of you that is illuminated by the headlights. You have an idea of where you would like your vehicle to be on the road over the next few seconds, but you have not worked out what your hands and feet need to do on the controls in order to take the corner safely and smoothly.

First, you take account of where your vehicle is in the context of the road, alongside its heading in comparison to your ideal path as well as its current velocity. This is analagous to the state vector in this implementation. You take this initial state and create a branching tree that moves forward in time from it, with many possible randomly generated sequences of inputs that all lead to different paths you could theoretically drive. You test each of these options by 'driving' the input sequences in your head and assigning a final score to each of them, with the paths that best lead you along your desired path scoring highly and the paths that take you away from it scoring poorly.

Using the score of each simulated input sequence as a coefficient, take a weighted average of each of these input sequences to condense it down into a single vector of inputs that is the length of your horizon. This vector is your new best guess of what your inputs will be for the next few seconds. However, and this is the important part of MPC and MPPI and all of the others, we only actually apply the first element of this derived vector and use that input for this current time step. After we move forward a single real-world time step, the assumption may no longer be valid, and it becomes necessary to redo this process over again, using our best guess as the new starting sequence to perturb and test new sequences upon.

There is a slightly more in-depth walkthrough of this MPPI implementation that includes the matrix transformations inside the python script.

### The Cost Function

Each simulation timestep (which is normally significantly smaller than your real world timestep), a measure is taken of the car's state relative to the desired path. A calculation is then made using these error factors and is added to an accumulator for that specific input sequence. It is important that this cost is calculated over the whole process of the simulation rollout and not just the final position of the vehicle, because that approach could technically result in input sequences that move sharply away from the desired path but subsequently come back around to end up in a good position receiving high scores. We want to reward closely following the raceline for the full duration of the rollout, not just on the final step of the horizon.

The actual cost function is denoted as
$$
    w_{\text{cte}} \lVert \mathbf{p}_k - \mathbf{p}^{\text{ref}}_k \rVert_2
    + w_{\psi} |\psi_k - \psi^{\text{ref}}_k|
    + w_{v} |v_k - v^{\text{ref}}_k|
    + w_{\delta} \delta_k^2
$$

Where the coefficients $w$ are tunable parameters that adjust how heavily each of the factors are rewarded/punished. 

$w_\text{cte}$ is the cross track error coefficient, or the lateral deviation from the desired position of the car at that point on the raceline.

$w_{\psi}$ is the heading error coefficient

$w_{v}$ is the desired velocity error coeffient

$w_{\delta}$ is a punishment for applying sharp steering. Including this produces noticeably smoother outcomes with less jitter.

### Other tunable parameters

The two main parameters for MPPI are the rollout count and the horizon. The rollout count is how many random sequences you generate and simulate at each timestep, and the horizon is how far into the future you test them. Both of these scale with the available compute. You generally want to ensure that your horizon is long enough to work through full corners before entry, and that you have a sufficient number of rollouts to discover a reasonably optimal path through them.

There is also a temperature parameter that informs how greedily you exploit your best found sequences. A high temperature denotes a large preference for the more highly scoring sequences in the final weighted average, while a low temperature takes a more balanced approach.

---

## Implementation

### Raceline

MPPI does not create an optimal raceline by itself around a track. Instead, it should be thought of as a highly effective controller that allows the vehicle to follow a precomputed raceline
as precisely as the internal model allows. In the case of onboard-compute limited operations like this one, the clear benefit is that the overall desired path is not being computed at runtime.
Instead, this extraordinarily computationally expensive process is offloaded to a stronger cluster before the main driving process even begins. The output of a pathline determining algorithm (see `raceline.csv`)
like Path Tracectory Optimization (PTO) contains a set of coordinate points that denote 'waypoints' on the optimal raceline that the car should follow, alongside helpful other pieces of data
such as the desired heading and speed at the associated point. While it is possible that these extra characteristics are somewhat cheap when compared to the overall path tracectory, it is more 
in keeping with the spirit of the challenge to move as much compute off-board as possible, and to create a lookup table of sorts with data that will be commonly referenced at runtime.

The yellow line in the simulation is a visualization of this precomputed raceline.

### Vectorised Rollouts

Because the rollouts are independent trials, they can be vectorised using NumPy (probably somewhat badly in this case due to my lack of experience with NumPy) to greatly improve the simulation rollout step, which is the computational bottleneck of this whole algorithm.

### Warm Start

There are two basic approaches to how you start each timestep and create your rollout input sequences. You can either completely forget everything about the past and fully randomly generate the sequences, or you can use a 'warm start' which uses your current best guess of where you want to go and creates small perturbations over that. In environments where conditions change rapidly (think of an exploring autonomous rock crawler moving purely on sensor input cresting a hill that it couldnot previously see over) and new context of the local state is discovereed every timestep, the scratch start strategy does have merit. However, in situations like racing where it is quite reasonable to expect your previous assumptions to hold true, it ends up being a far better use of computation to start with your previous best guess and make small deviations from it. The warm start approach is utilized in this implementation.

This current predicted control sequence is shown in the simulation in cyan.

---

## Results

The car accurately and smoothly follows the oval raceline. Take notice of the inital frames of the simulation where the car is facing the complete wrong direction: the car is able to turn and accelerate toward the desired direction before smoothly rejoining the line without heavy oscillation from side to side. 

---

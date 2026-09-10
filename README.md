# Wisconsin Autonomous: Controls Coding Challenge

Welcome! This is the coding challenge for the **controls team** of Wisconsin Autonomous. It is how we get to know how you think and program, and how you get a taste of what our work actually looks like. There are five challenges at different levels. **Pick one that fits your experience.**

Everything here is modeled on our real autonomy stack. We drive a drive-by-wire Chevy Bolt on ROS 2:

```
                                     perception (objects, lights, lanes)
                                                    |
 destination --> [ Route Planning ] --route--> [ Local Planning ] --+
                                                                    +--trajectory--> [ Trajectory Tracking ] --throttle, brake,--> car
 parking lot ----------------------------> [ Parking ] ------------+                 (Speed Control is its        steering
                                                                                       speed half)
```

## Pick your challenge

| Level | Challenge | You build | You will learn |
|---|---|---|---|
| Beginner | [**Route Planning**](route_planning/) | Dijkstra and A* on a town's road network, then rerouting around a closed road | Graphs and search |
| Beginner | [**Speed Control**](speed_control/) | PID cruise control for our car's throttle and brake: hold speed, climb hills, stop on the line | Feedback and feedforward control |
| Intermediate | [**Trajectory Tracking**](trajectory_tracking/) | A controller that steers and drives the car along the planned path | Vehicle geometry, pure pursuit, tuning |
| Intermediate | [**Local Planning**](local_planning/) | The planner that picks speed and position for the next 50 m: curves, stop signs, barrels | Planning with noisy perception |
| Advanced | [**Parking**](parking/) | A path that gets the car into a parking spot without touching anything | Kinematic path planning and search |

**Not sure?** If you have not taken a controls or robotics class yet, start with Route Planning or Speed Control. If you have, pick whichever sounds more fun. A beginner challenge done well and explained well beats an advanced one done badly: we are looking at how you think, not at which level you picked. (Want more? Do a second one.)

## How every challenge works

1. **A guided core.** You fill in a few small functions, one part at a time. Each part has a short explanation in the challenge README, and a checkpoint that tells you right away whether it works:
   ```bash
   python check.py part1
   ```
2. **Put it together.** `python run.py` drives every scenario, prints a scoreboard, and saves plots to `results/`.
3. **Stretch goals.** Open-ended extensions for people who get hooked. None are required.

Every part also has one or two **"explain in your own words"** questions. Your answers go in your write-up, and they matter as much as your code.

## Doing it yourself

Try each part on your own first. Use AI tools the way you would use a TA: to explain a concept, to help you read an error message, to get unstuck. Do not have one write your solution. You will learn more, and the checkpoints make it satisfying. Tell us in your write-up how you used AI tools; that is completely fine.

## Setup

You need Python 3.9 or newer. That's it: no ROS, no GPU, no simulator install.

```bash
git clone https://github.com/WisconsinAutonomous/Controls-Coding-Challenge.git
cd Controls-Coding-Challenge
python3 -m venv .venv                 # optional but recommended
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cd speed_control                      # or route_planning, trajectory_tracking, local_planning, parking
python check.py                       # every part starts out unfinished
python run.py
```

(Once the virtual environment is active, `python` means the right Python. Without one, use `python3` on macOS and Linux.)

Then open that challenge's README and start with Part 1.

## Repository layout

```
common/                shared by all challenges (message types, car parameters, geometry helpers)
route_planning/        beginner
speed_control/         beginner
trajectory_tracking/   intermediate
local_planning/        intermediate
parking/               advanced
```

In each challenge you only edit the one file its README tells you to (plus any new files of your own). The simulators in `sim/` are what we grade with, so changes there will not help you.

## What to submit

Create a **public GitHub repository** (a fork of this one is fine) that contains:

1. **Your code.** It must run with the unmodified `run.py` and `check.py` from this repo.
2. **Your `results/` folder** from a final `python run.py`: the plots, `summary.json` and `scoreboard.md`.
3. **A write-up** (`WRITEUP.md` in the repo root, about 1 to 2 pages) with:
   * your answers to the "explain in your own words" questions, each with the plot that shows it,
   * what did not work at first and how you figured it out,
   * the challenge-specific items in that challenge's README,
   * what you would do with another week,
   * how you used AI tools.

Then submit the link to your repository through the application form linked from our recruiting post.

## How we evaluate

| | What we look at |
|---|---|
| **Checkpoints and results** | How far through the parts you got, and your scores on the public scenarios and on held-out ones (different random seeds). |
| **Understanding** | Your write-up: do you know why your solution works and where it breaks? |
| **Code** | Is it readable and organized? Could a teammate pick it up next semester? |

It is completely fine to not finish everything. An honest, well-explained attempt at the core is a good application.

## Questions

If something in the challenge itself is broken or unclear, open an issue on this repository.

Good luck, and have fun!

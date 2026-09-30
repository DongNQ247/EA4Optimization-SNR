# Genetic Algorithm for Optimization of Signal-to-Noise Ratio (SNR)

## Abstract

This repository studies an impedance-optimization problem that maximizes the signal-to-noise ratio (SNR) in a communication system. Closed-form analytical expressions are derived, and the problem is reformulated as a constrained optimization task. A Genetic Algorithm (GA) is implemented to validate the analytical solution under identical impedance constraints.

## Introduction

The signal-to-noise ratio (SNR), defined as the ratio of received signal power to noise power, is a key performance metric in communication systems. Adjusting the receiver and load impedances, denoted by $Z_R$ and $Z_L$, directly influences both signal transfer and noise generation.

The SNR model used in this work is

$$
\mathrm{SNR}
= \frac
{
g^2\left|\dfrac{Z_L}{Z_R + Z_L}\right|^{2}
\lvert Z_{RT} \rvert^{2} S_{IT}
}
{
N_{na} + g^2\left|\dfrac{Z_R}{Z_R + Z_L}\right|^{2} 2 k T\, \Re\{Z_L\}
}
$$

where $g$ is the amplifier gain, $Z_{RT}$ is the transfer impedance between the transmitter and the receiver, $S_{IT}$ is the input signal power, $N_{na}$ represents additive noise, $k$ is Boltzmann’s constant, and $T$ is the absolute temperature.

The design objective is to select $Z_R$ and $Z_L$ to maximize SNR subject to physical limits on their real and imaginary components; specifically, we consider maximizing SNR with $\Re\{Z_R\}, \Re\{Z_L\}$ and $\Im\{Z_R\}, \Im\{Z_L\}$ constrained to the ranges stated below.

$$
\begin{aligned}
\max_{Z_R,\,Z_L} \quad& \mathrm{SNR}(Z_R, Z_L) \\
\text{s.t.} \quad
& \Re\{Z_R\} \in [R_{R,\min}, R_{R,\max}], \quad R_{R,\min} > 0 \\
& \Re\{Z_L\} \in [R_{L,\min}, R_{L,\max}], \quad R_{L,\min} > 0 \\
& \Im\{Z_R\} \in [-X_R, X_R], \quad X_R \ge X_L \\
& \Im\{Z_L\} \in [-X_L, X_L], \quad X_L > 0
\end{aligned}
$$

## Optimization Using Genetic Algorithm

### Experimental Setup

A Genetic Algorithm is employed to maximize the same SNR function under identical impedance constraints. The constants used in the experiment are:

We set the following constants:
$g=5.0$, $Z_{RT}=50\Omega$,
$S_{IT}=1W$, $N_{na}=1\times10^{-9}$,
$k=1.380649e-23$, and $T=290K$.
Consequently, we set $\alpha = N_{na} = 1\times10^{-9}$ and
$\beta = g^{2}2kT=1.656e-20$.
The feasible bounds are specified as follows

$$
\begin{aligned}
\Re\{Z_R\}\!\in[10,10000], \quad \Im\{Z_R\}\!\in[-10000,10000], \\
\Re\{Z_L\}\!\in[10,10000], \quad \Im\{Z_L\}\!\in[-10000,10000],
\end{aligned}
$$

### Parameters

Each chromosome is encoded as a four-dimensional real vector

$$
[\Re\{Z_R\},\, \Im\{Z_R\},\, \Re\{Z_L\},\, \Im\{Z_L\}],
$$

GA configuration:

- Population size: $N=20$
- Generations: $G=1500$
- Crossover: SBX, $\eta_c=10$, $p_c=0.8$
- Mutation: Polynomial, $\eta_m=10$, $p_m=0.2$
- Selection: Tournament (size = 4)
- Survival selection: Elitism

## Directory architecture

```text
├── README.MD
└── src
    ├── algorithm.py  # Algorithm implementation
    ├── individual.py # Definition of individual
    ├── main.py       # Main program
    ├── populution.py # Definition of population
    └── problem.py    # Problem definition
```

## Result

![](result/result.png)

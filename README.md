# Oikos: Closed Loop Ecosystem Simulator in C



An ecosystem simulator, from plate tectonics and weather down to the redox chemistry of individual microbial metabolisms, with animals whose behaviour is learned by recurrent neural networks. It is written in plain C99 with no dependencies. Every exponential, logarithm, trigonometric function, random number generator, tensor operation and gradient is implemented in the source.

## How the Earth works

The Earth absorbs sunlight as photons. These photons can move electrons from one molecule to another, storing potential energy in chemical bonds. Plants take advantage of this by making sugar through photosynthesis. Organisms get energy back from sugar by passing electrons from sugar to oxygen. Moving turns the energy into heat almost right away. Growing stores it in their bodies until they're eaten or decompose, and then it becomes heat too, which all eventually radiates out into space. This leaves behind low-energy molecules, carbon dioxide and water, which plants can use to store energy from sunlight again, and the cycle repeats.

---

## Existing ecosystem simulations and their limits

Existing ecosystem models tend to excel at one scale and simplify the others.

| Family | Examples | What they do well | What they usually leave out |
|---|---|---|---|
| Earth system / biogeochemical models | CESM, ERSEM, Darwin (MIT), LPJ and other DGVMs | Realistic climate, ocean and nutrient cycles at global scale | Organisms are fixed functional types; no evolution or behaviour; mass balance is maintained per module rather than proven per reaction |
| General ecosystem models | Madingley (Harfoot et al. 2014), Ecopath with Ecosim | Mechanistic feeding, metabolism and size structure across all animals | Microbes and chemistry are reduced to a nutrient pool; climate is an input, not simulated |
| Artificial life and evolution | Tierra, Avida, Polyworld, Framsticks | Open-ended evolution and emergent behaviour | No physical chemistry or energy budget; "food" is a token |
| Agent-based teaching models | NetLogo Wolf-Sheep, many game simulators | Visual predator-prey dynamics | Energy is created from nothing; no element cycles; populations persist only by tuning |
| Reinforcement learning environments | Neural MMO, grid-world foraging | Learned behaviour in multi-agent settings | The world has no ecology: resources respawn rather than cycle |

There are three recurring gaps in these models. Matter is not strictly conserved, so a model can quietly create or destroy carbon and still look healthy. Metabolism is not tied to thermodynamics, so organisms grow on reactions that would yield no energy. And evolution, behavior and the physical environment live in separate codebases, meaning none affects the others.

---

## This simulator

### Strict conservation
All state changes are expressed as `Transfer`s between pools. Each species, mineral, organic class, organism and gas carries a composition vector over **11 conserved elements** (C, N, P, S, Fe, Ca, Na, K, Mg, Cl and electrons). A transfer is rejected unless it balances every component, including electrons, to 1 part in 10¹² and leaves no pool negative. Pool totals use compensated summation, and a positivity limiter scales competing demands on the same pool so that no process can draw more than exists.

### The planet
- **Geometry**: an icosphere subdivided up to four times (12 to 2,562 cells), each with water layers, soil or sediment layers, and a lithosphere reservoir.
- **Tectonics and terrain**: random plates with rotation vectors give convergence, divergence and shear at boundaries. Simplex-noise terrain is blended from eight landform shapers (ridges, plains, terraces, cliffs and others) by a learned-style weighting of uplift, wetness, slope and hardness, then carved by stream-power erosion with sediment deposition and hillslope creep.
- **Hydrosphere**: oceans and lakes are found by a depression-filling solver that spills water from basin to basin; soils are typed by sediment, wetness and slope.
- **Geology**: earthquakes (Gutenberg–Richter magnitudes, subduction depths) and eruptions (VEI from arc, ridge and hotspot sources) are generated deterministically by hashing, so any epoch can be queried without replaying history.

### Climate and hydrology
An energy-balance atmosphere with a diurnal and seasonal Sun, linearised outgoing longwave radiation, lapse rates, clouds from humidity, snow and sea-ice albedo. Winds combine Hadley, Ferrel and polar cells, a thermal-wind correction and drifting simplex eddies. Water vapour is advected and diffused between cells, condenses into rain or snow, infiltrates and drains through soil, runs off towards basins, creates lakes, overflows to the sea and changes ocean depth. The climate is simulated for a year before adding in organisms.

### Chemistry
22 dissolved species, 7 gases, 8 minerals and 6 classes of particulate organic matter. pH is solved from charge balance with temperature-dependent carbonate, ammonia, sulfide, borate, phosphate and acetate equilibria. Gases exchange with the atmosphere through Henry's law. Phosphate sorbs onto iron hydroxide and calcite. Sulfide and ferrous iron oxidise, calcite and iron sulfide precipitate and dissolve, all scaled by their Gibbs energy so reactions stop at equilibrium. Detritus hydrolyses, refractory matter photodegrades, and sediments are buried and slowly returned.

### Microbes
19 functional types (oxygenic and diazotrophic phototrophs, heterotrophs, nitrifiers, denitrifiers, iron and sulfur oxidisers and reducers, methanogens, methanotrophs, fermenters and others) built from 16 redox half-reactions. Growth is solved thermodynamically: a cohort's electron carrier settles at the potential where donor and acceptor fluxes balance, each flux falls to zero as its free-energy gap approaches a minimum energy quantum, and growth is limited by captured power after maintenance, Heijnen synthesis costs, Monod substrate uptake and Droop-style variable N and P quotas. Energy shortfalls burn biomass by respiration or fermentation.

Each cohort carries a 21-locus genome with trade-offs (growth against affinity, thermal breadth against peak, pigment against maintenance). Mutations include point and large effects, knockouts, gene duplication, promoter capture and multi-step potentiation (so key innovations such as degrading refractory matter can arise, as in Lenski's citrate experiment), mutator alleles and neutral passengers. Mutants establish with branching-process probabilities. Small populations die stochastically; cohorts can go dormant, attach as biofilm, disperse by mixing, sinking or air, and exchange genes by transformation, transduction and conjugation.

### Plants, animals and pathogens
- **Plants**: grass, shrub and tree cohorts with light-use efficiency, temperature, soil-water, nutrient and CO₂ limits, respiration and litter.
- **Animals**: eleven types from microzooplankton to large land mammals, feeding by size-selective functional responses with interference, limited by stoichiometry, and killed by hazards from pH, low oxygen, sulfide, ammonia and temperature. Some make resting eggs that hatch on environmental cues.
- **Pathogens**: viruses with lytic and lysogenic cycles, burst size, host range and induction under starvation; animal parasites spread through free stages, contact, faeces, eggs and spores. Their traits evolve.

### Animals that learn
Every animal cohort is a **body** controlled by a brain shared across its species. Each brain is a recurrent actor-critic: SwiGLU encoders over a size-spectrum view of food and predators, the body's own state and up to eight neighbouring places; GRU memory; and four action heads (diet size, feeding effort, egg laying, and a pointer-attention head choosing where to move). Brains are trained with PPO and generalised advantage estimation over truncated chunks of experience, all on the built-in automatic differentiation engine.

Evolution is the last layer of this simulation. Each body inherits a small "temperament" network and four genes (planning horizon, curiosity, motivation, budding share) that shape its intrinsic rewards and action priors. Bodies bud daughters when they grow, mutate on inheritance, and are archived by how many adult offspring they produced; new bodies draw from that archive. Learning tunes behavior within a lifetime; evolution tunes what is worth learning.

### Numerics and engine
Time stepping is adaptive Heun (predictor–corrector) with split transport half-steps and step-size control from the predictor error. The general-purpose engine underneath includes double-double precision maths, PCG random streams, an N-dimensional tensor library, a tape-based autograd with recurrent (delayed) edges, convolution, normalisation and loss layers, and SGD, Adam, Muon and ANO optimisers.

---

## Output

A plain run prints one row per day: mean surface pH, O₂, dissolved inorganic carbon, ammonium, nitrate, phosphate and temperature; carbon in phototrophs, other microbes, plants and dormant cells; animal numbers in water and on land; and counts of microbial cohorts and animal bodies. Training runs print per-species reward, policy and value losses, entropy, KL divergence and clipping statistics, and save brains and genome archives to the file given.

## Status and caveats

This is a research environment, not a forecasting model, and I have a lot of other items to complete before I would consider this finished. Rates and constants are drawn from the literature but simplified, and the default planet (162 cells, 10-minute maximum steps) is chosen to run on a single CPU core. Larger planets and long training runs are slow. Contributions, bug reports and validation against real ecosystems are welcome.

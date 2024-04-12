 # Reflect Orbital Solar in Space Performance Sim

<div align="center">
  <img src="media/orbit.gif" alt="orbit"/>
</div>

## Simulation

The `simulate()` function outputs 3 tensors:
* `optimal_transmission`: the transmission factor from the sun to each farm at any given point of the year and at a given orbit. 
* `optimal_farms`: the index of the farm chosen for the optimal transmission rate, a given point in time and a given orbit.
* `cumulative_farm_allocation`: the maximum amount of energy that each farm could receive from a single satellite in a year of operation. Tensor dimensions run through all the different simulated orbits.

Pretty much everything should be configured in `settings.py`.

### Fast and easy

Click [here](https://drive.google.com/file/d/1uKxBfArj3dqMb1flkqKNwYFkRO3NUJQO/view?usp=sharing) to download the result of the simulation with the default parameters (~17GB) and place in the `out/` directory. 

To tune the parameters and re-run the sim, follow the steps in the next sections.

### Pre-requisites

* ```sh
  git clone https://github.com/Popeyef5/reflect_orbital.git
  pip install -r requirements.txt
  ```
        
* [PyTorch](https://pytorch.org/get-started/locally/)

* A [Modal](https://modal.com) account (preferred) for cloud GPU usage. The free tier is enough for plenty of sims

### Run locally

```sh
python simulation.py
```

### Run remotely (recommended)

setup and log in a [Modal](https://modal.com) account. Then: 
```sh
modal run -d cloud.py
```
followed by
```sh
modal volume get reflect-orbital results.safetensors out/
```
the downloaded file will weigh ~17GB with default settings. Make sure it ends up in the `out/` directory as intended.

## Analisis

```sh
python analysis.py
```
will generate the figures currently in `media/`. Easy to tune or customize. Importantly, everything was calculated with 0 opex, which is of course a bit unrealistic. Also, with so many parameters, one has to be aware that most figures varying one or two chose a specific value for all others. Figures revolve around some interesting metrics:

* `Interest rate analisis`: under different circumstances, what is the highest APR on a fixed interest loan that one could take to fund the operation so as to breakeven at the end of the satelite lifetime. Determined as a function of satellite cost, opex and annual revenue.

* Raw power output in the best performant orbit (although performance varies little)

* `Farm cumulative allocation` and `number of compatible farms `: essentially, how much power can be potentially delivered to each individual farm by a single satellite (overlaps included, so not contemplating that giving 100% of this to each is impossible). Combined with the setting `MIN_`, we can get a hand-wavy and somewhat overly-optimistic estimation of the number of farms that could be served an amount of power greater than the minimum required as a function of the number of satellites in the constellation. 

### Further analisis

Some interesting further research ideas:

* Combine the result of `number of compatible farms` with its implication on customer % and the results on `interest rate analisis`: since a constellation of a single satellite in orbit woudl yield negligible revenue for farms, getting them on board would be tough. On the contrary, with a gigantic constellation, a lot of farms would be eligible as clients. Essentially, analyze the effects of going to scale and how big of a constellation one would need to get to a certain customer % coverage. Then from this infer the max tolerable APR. Although this might not be the avenue for financing the constellation, it gives a ballpark for the economic ROI of the satellites.

* Mask the `number of compatible farms` to consider different types of farms (operating, construction, etc...) and incorporate the project name in the analysis. This would enable the study of revenue generated vs CAPEX which could answer an important question: How do farms potentially come online as customers as the constellation size increases and how does revenue grow with it? How much does this cost in terms of raw CAPEX? 

* Modify sim to get this optimally considering the effects of farm availability superposition.

* While Starthip might be a fair bet, it might be interesting to analyze how the results vary with different values for `KG_COST_TO_ORBIT`

* Managing the raw `sun_farm_transmission` variable as is is impossible since it's a few TB. However picking the best orbit with the current settings and seeing that the difference in performance compared to other orbits is not huge, a good analysis would be to reduce the amount of `POINTS_PER_DAY` to see if we can manipulate the raw variable and do some more complex analisis.

* Analyze how many farms the optimal orbit interacts with and plot the contribution of each. It's possible that the contribution follows a pareto tail and small number of big farms contribute most of the revenue, which would be sort of an ideal case because less customer onboarding would be needed and they usually have little area losses.

Not included because in principle because they might be less important than the financial variables but straightforward to calculate:

* LCOE
* Annual ROI
* Total energy beamed down per year
* etc

hit gaston@myelin.vc with comments or bugs
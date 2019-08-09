Network Structure
************************************

* The study includes four types of networks: random, community-structured, scale-free (power law), and an empirical degree distribution. 
* The population size is 5,000 individuals. 
* Assumed the same average number of degree across types of network in the simulation time horizon. 
* Time step is bi-weekly time step. 
* Simulate the network and disease transmission dynamics at two levels of disease prevalence: 5% and 20%. 
	* The disease prevalence is the steady state prevalence given the standard partner management (partner notification). 
* Different levels of disease prevalence required different time horizon and analysis time: 

.. list-table:: **Target prevalence with PN**
  :widths: 10 10 10 10 10
  :header-rows: 1

  * - Prevalence
    - Timehorizon
    - Strategy implemented in the burn-in period
    - The initial time step a strategy is implemented
    - Analysis window
  * - ``5%``
    - 50 years
    - Annual screening alone
    - At the 30th year
    - Last 10 year of the simulation
  * - ``20%``
    - 20 years
    - Annual screening alone
    - At the 8th year
    - Last 10 year of the simulation

==================
Random networks
==================

    .. raw:: html

		<img src="fig/random.png" style="width: 80%">

		<img src="fig/random dd cc duration (correlated).eps" alt="random network output distribution"  style="width: 100%">


====================================
Community-structured networks
====================================

    .. raw:: html

		<img src="fig/community.png" style="width: 80%">

		<img src="fig/community dd cc duration (correlated).eps" alt="community network output distribution"  style="width: 100%">


====================================
Scale-free networks
====================================

    .. raw:: html

		<img src="fig/power_law.png" style="width: 80%">

		<img src="fig/power_law dd cc duration (correlated).eps" alt="power_law network output distribution"  style="width: 100%">



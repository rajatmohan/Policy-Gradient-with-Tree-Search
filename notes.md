## standard actor-critic + ppo setup
1) initialize actor network (policy)
2) initialize critic network (value)
3) collect batch of trajectories (you can have a fixed length of trajectory or leave it till the episode ends).
4) calculate rewards to go (monte carlo rewards - RTG)
5) calculate advantage (A(s,a) = RTG - V(s)). here V(s) comes from critic network. store (state, action, log prob, reward, value of state, RTG)
6) optimize critic network (use mse loss = (RTG - V)) 
7) optimize actor network (use ppo objective)
$L(\theta) = E_{\theta}[min(r_t(\theta)A_t, clip(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t)]$. here we have to max this objective. need gradient ascent. pytorch handling use: $-L(\theta)$
8) repeat step 6 and 7 n times (what is n?)
9) repeat step 3 to 8 T times



## our pgts setup
4) the agent branches out $m$ steps into the future, asks the critic for its evaluation at the leaves, and calculates $T^m V(s)$ (use the lagging policy to ask for future)
5) A(s,a) = $T^m V(s)$ - V(s)
6) mse loss = ($T^m V$ - V)
8) right after doing step 8) n times, update the lagging policy 

---

- a lagging policy is simply a carbon copy of your Actor network. but instead of updating instantly after every batch, it updates slowly using an exponential moving average (EMA).
- also, use pure mean to sample while recording video
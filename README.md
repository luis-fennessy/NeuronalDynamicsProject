# NeuronalDynamicsProject
Main Project for NX_465 - Hopfield networks under biological constraints

### Exercise 0 Note to self:

I had some trouble with ex0.2 and ex0.3 - I often saw the two 'new state' outcomes to be different for the same initial patterns and starting state. eg:

```
N, P = 4, 2
self.patterns = np.array([[ 1, -1,  1, -1],
                                    [-1,  1,  1, -1]])
self.state = np.array([-1, -1,  1,  1])
```

Returns two different outcomes for the two exercises even though they're meant to be equivalent methods:

```
[ 1.  1. -1. -1.]
[1. 1. 1. 1.]
```

In short, the 0.3 method returns an array of 1s because the ```m``` array is all zeros, which means the next state is ```sign([0, 0, 0, 0])``` which we define as ```[1,1,1,1]```
The 0.2 method produces the following weights:
```
[[ 0.  -0.5  0.   0. ]
 [-0.5  0.   0.   0. ]
 [ 0.   0.   0.  -0.5]
 [ 0.   0.  -0.5  0. ]]
```
and ultimately returns the next state ```[-1 -1  1  1]```. I'm equally confident that this method is performed correctly.

I looked a little into why this is possible. **The overlap method (Eq. 3) is an approximation unless patterns are orthogonal or P<<N**. In our case, P is too close to N

- The overlap-based method approximates the Hebbian sum by factoring out p{mu}{i}
- This approximation is not exact, even with orthogonal patterns
- It gets better as N -> infinity

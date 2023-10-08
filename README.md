# Hypersonic Golf Ball Trajectory (with a bad model)

Uses Runge Kutta 4 to estimate the trajectory of a sphere with a drag coefficient similar to a golf ball.

We also estimate the optimal launch angle for maximizing distance traveled and fit a model to the angle/distance curve to extrapolate the predictions.

![Test 1: Trajectory Visualization](/output/test_1.png)
Here we visualize the parabolic trajectory with no air resistance, the simulated trajectory with a 45 degree angle, and the trajectory of a launch optimized for maximum distance given a set velocity.

All balls were launched with an initial velocity of 50 m/s.

![Test 2: Angle/Distance Curves](/output/test_2.png)
In order to get a more detailed view of the relationship between the launch angle and total distance traveled, we plot the relationship between the two at varying initial launch velocities. We can see clearly that as the initial velocity increases, the function skews further and further right causing the optimal launch angle to fall at smaller and smaller values.

![Test 3: Optimized Trajectory Visual](/output/test_3.png)
This is just a nice visualization of the optimal trajectories for a range of initial velocities. It is easy to see the slowly reducing angle as the velocity increases.

These next three visualizations all show the relationship between launch velocity and optimal angle. Out of a polynomial fit, logarithmic fit, exponential fit, and 1/x fit, the 1/x model is by far the most accurate, although as we will analyze, it is not perfect and does not capture the true dynamics. However, at all reasonable speeds it is a very good model.

To be accurate, the model we use is:
$$\frac{a}{x+b} + c$$

This is a reasonable guess for a model since we appear to have very high slopes near zero which evens out and becomes almost flat eventually. A logarithmic model would catch the high initial slope and an exponential model would capture the eventual constant value, but this nonlinear model performs the best at both extremes.

![Test 5: Launch angle vs initial velocity - Realistic range (10m/s to 100m/s)](/output/test_5.png)
![Test 4: Launch angle vs initial velocity - Moderate range (10m/s to 300m/s)](/output/test_4.png)
![Test 6: Launch angle vs initial velocity - Large range (10m/s to 10000m/s)](/output/test_6.png)

Now, since our ball is moving at mach 30, our simple aerodynamical model likely breaks down... along with the golf ball which is now a shower of molten plastic. However, it is still interesting to see what the result would be in such extreme environments.

Model failures:
1. The optimal angle near 0 does not in fact become large. It actually stays at 45 degrees. This is not well captured by our model which is convex everywhere (in our domain).
2. The sum of squared residuals increases as our range of velocities does. This is indicative of the fact that our model does not actually capture the true underlying generating equation which is a non-elementary function.

![Test 8: Initial Velocity vs Max Height (10m/s to 10000m/s)](/output/test_8.png)
As we can see, the height achieved by golf balls hit at hypersonic speeds still is not that high. This is due to the low mass mixed with the very high resistance at large speeds.

![Test 9: Trajectories with very high mass (100m/s to 10000m/s)](/output/test_9.png)
If we make this ball out of a much... much heavier material. We can see it is not nearly as effected by the atmospheric drag. In fact, it's getting close to escaping the atmosphere entirely which means that our assumption of constant atmospheric density is messing everything up.

Anyways, this was fun. Hope you got what you needed. Oh, here's full trajectories for very hypersonic golf balls.

![Test 7: Launch Trajectories - Large range (100m/s to 10000m/s)](/output/test_7.png)

TODO:
- [ ] Implement varying atmospheric density due to height
- [ ] Implement Magnus force and expand simulation to 6DOF
- [ ] Implement drag correlations with Re number

Will I ever do these things? Maybe. Seems like fun.
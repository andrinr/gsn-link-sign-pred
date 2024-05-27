# Neural ODE

We have the adjoint 

$$\mathbf{a}(t) = \frac{\partial L}{\partial \mathbf{z}(t)}$$

and we are given the dual ODE

$$\frac{d\mathbf{a}(t)}{dt} = -\mathbf{a}(t) \frac{\partial f(\mathbf{z}(t), t, \theta)}{\partial \mathbf{z}}$$

where $f$ is the neural ODE function.
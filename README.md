# SpringE: Fast Node Representation Learning for Signed Networks

Tasks in network analysis often involve predicting on a node, edge or
graph level. Signed networks, such as social media networks, con-
tain signs that indicate the nature of the relationship between two
associated nodes, such as trust or distrust, friendship or animos-
ity, or influence or opposition. In this paper, we propose SpringE, a
node representation learning algorithm for link sign prediction with
comparable performance as graph neural network based methods.
SpringE directly models the desired properties as an energy gradi-
ent using a physics-inspired spring network simulation based on as-
sumptions from structural balance theory which can be solved using
standard numerical integration methods for ODE.
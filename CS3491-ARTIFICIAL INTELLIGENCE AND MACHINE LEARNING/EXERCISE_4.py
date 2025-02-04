from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Step 1: Define the structure of the Bayesian Network
model = BayesianNetwork([('Rain', 'Grass'), ('Sprinkler', 'Grass')])

# Step 2: Define the Conditional Probability Distributions (CPDs)

# P(Rain) - Prior probability for Rain
cpd_rain = TabularCPD(variable='Rain', variable_card=2, values=[[0.7], [0.3]])

# P(Sprinkler) - Prior probability for Sprinkler
cpd_sprinkler = TabularCPD(variable='Sprinkler', variable_card=2, values=[[0.6], [0.4]])

# P(Grass | Rain, Sprinkler) - Conditional probability table for Grass
cpd_grass = TabularCPD(
    variable='Grass', 
    variable_card=2, 
    values=[[0.99, 0.9, 0.8, 0.0],  # P(Grass=False)
            [0.01, 0.1, 0.2, 1.0]],  # P(Grass=True)
    evidence=['Rain', 'Sprinkler'], 
    evidence_card=[2, 2]
)

# Step 3: Add CPDs to the model
model.add_cpds(cpd_rain, cpd_sprinkler, cpd_grass)

# Verify the model is correct (CPDs should sum to 1)
assert model.check_model()

# Step 4: Perform inference
inference = VariableElimination(model)

# Query 1: What is the probability that the Grass is wet (True) given that it rained?
result_1 = inference.query(variables=['Grass'], evidence={'Rain': 1})
print("P(Grass=True | Rain=True):\n", result_1)

# Query 2: What is the probability of rain given that the Grass is wet and the sprinkler is on?
result_2 = inference.query(variables=['Rain'], evidence={'Grass': 1, 'Sprinkler': 1})
print("P(Rain | Grass=True, Sprinkler=True):\n", result_2)

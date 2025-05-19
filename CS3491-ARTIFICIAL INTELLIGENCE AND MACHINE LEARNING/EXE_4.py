# Import necessary libraries
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define the structure of the Bayesian Network
model = BayesianModel([('Burglary', 'Alarm'),
                         ('Earthquake', 'Alarm'),
                         ('Alarm', 'JohnCalls'),
                         ('Alarm', 'MaryCalls')])

# Define CPDs for each node

# Prior for Burglary (0: No burglary, 1: Burglary)
cpd_burglary = TabularCPD(variable='Burglary', variable_card=2,
                          values=[[0.999], [0.001]])

# Prior for Earthquake (0: No earthquake, 1: Earthquake)
cpd_earthquake = TabularCPD(variable='Earthquake', variable_card=2,
                            values=[[0.998], [0.002]])

# CPD for Alarm given Burglary and Earthquake
# The values are arranged for the combinations: [Burglary=0, Earthquake=0], [0,1], [1,0], [1,1]
cpd_alarm = TabularCPD(variable='Alarm', variable_card=2,
                       values=[[0.999, 0.71, 0.06, 0.05],   # Alarm = 0 (False)
                               [0.001, 0.29, 0.94, 0.95]],  # Alarm = 1 (True)
                       evidence=['Burglary', 'Earthquake'],
                       evidence_card=[2, 2])

# CPD for JohnCalls given Alarm
cpd_john = TabularCPD(variable='JohnCalls', variable_card=2,
                      values=[[0.95, 0.10],  # John does not call
                              [0.05, 0.90]], # John calls
                      evidence=['Alarm'], evidence_card=[2])

# CPD for MaryCalls given Alarm
cpd_mary = TabularCPD(variable='MaryCalls', variable_card=2,
                      values=[[0.99, 0.30],  # Mary does not call
                              [0.01, 0.70]], # Mary calls
                      evidence=['Alarm'], evidence_card=[2])

# Add CPDs to the model
model.add_cpds(cpd_burglary, cpd_earthquake, cpd_alarm, cpd_john, cpd_mary)

# Validate the model structure and CPDs
assert model.check_model(), "Model invalid: Check CPDs and structure."

# Perform inference using Variable Elimination
infer = VariableElimination(model)

# Example Query:
# Calculate the probability of a burglary given that both John and Mary called (1 indicates 'True')
query_result = infer.query(variables=['Burglary'], evidence={'JohnCalls': 1, 'MaryCalls': 1})
print(query_result)
